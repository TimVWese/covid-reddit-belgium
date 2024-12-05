using Optim
using RollingFunctions

include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[]), date_range=nothing)

subreddits = ["belgium"]
topics = [lockdown, mask, vaccin]
path = joinpath(RESULT_DIR, "02_timeseries")
start_date = Date(2020, 1, 1)
end_date = Date(2022, 12, 31)
windowsize = 14

negative_day_thres = 50 # number of posts before a negative day is considered
negative_day_quantile = 0.28 #quantile to use for negative day detection

keydates = Dict([
    vaccin => DataFrame([
        :date=>[Date(2020,3,16),Date(2020, 12, 28), Date(2021, 9, 22), Date(2021, 11, 01), Date(2022, 07, 01)],
    ]),
    mask => DataFrame([
        :date=>[Date(2020,07,09), Date(2021,9,17), Date(2021,11,17), Date(2022,03,04)],
    ]),
    lockdown => DataFrame([
        :date=>[Date(2020, 3, 13),Date(2020, 6, 8),Date(2020,7,29),Date(2020, 08, 26),Date(2020, 10, 19),Date(2021,6,9), Date(2021,11,27), Date(2022,02,18)],
    ])
])

"""
    per_day(df::DataFrame, ops::Pair...)

Aggregate per day, aggregating the desired columns by the `ops`. 
"""
function per_day(df::DataFrame, ops::Pair...)
    pd = groupby(df, :date) |>
          group -> combine(group, :id => length, ops..., renamecols=false) |>
                   df -> sort(df, :date)
    DataFrames.rename!(pd, :id => :num)
    result = DataFrame(:date=>minimum(pd.date):Day(1):maximum(pd.date))
    leftjoin!(result, pd, on=:date)
    return coalesce.(result, 0)
end

"""
    rolling_per_day(comments; selector=row->true, windowsize=14)

Get the rolling mean of the number of comments per day
"""
function rolling_per_day(comments; selector=row->true, windowsize=14)
    pd = per_day(comments[[selector(row) for row in eachrow(comments)],:])
    pre_pad = windowsize ÷ 2
    post_pad = (windowsize - 1) ÷ 2
    pd.total = vcat(fill(0, pre_pad), rollmean(pd.num, windowsize), fill(0, post_pad))
    return pd[pre_pad:end-post_pad-1, :]
end

function get_activity(comments, submissions, topic, subreddit; windowsize = 14, start_date=Date(2020, 1, 1), end_date=Date(2022, 12, 31))
    selector=row->row.topic==topic && row.subreddit==subreddit
    comment_act =  rolling_per_day(comments; selector, windowsize)
    submission_act = rolling_per_day(submissions; selector, windowsize)
    activity = DataFrame(date=start_date:Day(1):end_date)
    leftjoin!(activity, comment_act, on=:date, makeunique=true)
    leftjoin!(activity, submission_act, on=:date, makeunique=true)
    rename!(activity, :total => :nb_coms, :total_1 => :nb_subs)
    activity.nb_posts = coalesce.(activity.nb_coms, 0.) .+ coalesce.(activity.nb_subs, 0.)
    return activity
end

struct PiecewiseLinear
    knots::Vector{Date}
    values::Vector{Float64}
end

function (p::PiecewiseLinear)(x::Date)
    i = searchsortedlast(p.knots, x)
    if  i < length(p.knots)
        return p.values[i] + (p.values[i+1] - p.values[i]) * (x - p.knots[i]).value / (p.knots[i+1] - p.knots[i]).value
    else
        return p.values[end]
    end
end

function construct_loss(knots, data)
    return x -> begin
        pl = PiecewiseLinear(knots, x)
        sum((pl.(data.date) .- data.nb_posts).^2)
    end
end

function get_trend(data, knots; start_date=Date(2020, 1, 1), end_date=Date(2022, 12, 31))
    drange = max(start_date, minimum(data.date)):min(end_date, maximum(data.date))
    data = data[ein(data.date, drange), :]
    knots = [drange[1], knots..., drange[end]]
    init_vals = [data.nb_posts[findfirst(data.date .== d)] for d in knots]
    loss = construct_loss(knots, data)
    result = optimize(loss, init_vals)
    return DataFrame(date=knots, vals=Optim.minimizer(result))
end

function identify_negative_days(comments, topic, subreddit; language="en", sentiment_col=:bert, q=.25, threshold=50)
    lower_bound = x -> quantile(x, q)
    selection = comments[comments.topic .== topic .&& comments.subreddit .== subreddit .&& select_lang(comments, language), :]
    valid = remove_invalid_rows(selection, [sentiment_col, ])
    valid.w_sent = valid.score .* valid[!, sentiment_col]
    w_sent_lb = lower_bound(valid[!, :w_sent])
    pd = per_day(valid, :w_sent => median)
    return pd.date[pd[!,:w_sent] .< w_sent_lb .&& pd.num .> threshold]
end

isdir(path) || mkdir(path)

for subreddit in subreddits
    for topic in topics
        pd = get_activity(comments, submissions, topic, subreddit; start_date, end_date, windowsize)
        CSV.write(joinpath(path, "$(topic)_$(subreddit).csv"), pd[!,[:date, :nb_posts]])
        trend = get_trend(pd, keydates[topic].date; start_date, end_date)
        CSV.write(joinpath(path, "$(topic)_$(subreddit)_trend.csv"), trend)
    end
end

open(joinpath(path, "negative_days.csv"), "w") do f
    for subreddit in subreddits
        for topic in topics
            neg_days = identify_negative_days(comments, topic, subreddit; q=negative_day_quantile, threshold=negative_day_thres)
            println(f, "$subreddit,$(topic): "*join(neg_days, ", "))
        end
    end
end
