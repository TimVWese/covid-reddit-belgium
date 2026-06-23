using GLM
using Distributions
using RollingFunctions
using Printf

include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[]), date_range=nothing)

subreddits = ["belgium"]
topics = [lockdown, mask, vaccin]
path = joinpath(RESULT_DIR, "02_timeseries")
start_date = minimum(DATE_RANGE)
end_date = maximum(DATE_RANGE)
windowsize = 14

negative_day_thres = 50 # number of posts before a negative day is considered
negative_day_quantile = 0.275 #quantile to use for negative day detection

"""
    rolling_per_day(comments; selector=row->true, windowsize=14)

Get the rolling mean of the number of comments per day
"""
function rolling_per_day(comments; selector=row->true, windowsize=14)
    pd = per_day(comments[[selector(row) for row in eachrow(comments)],:])
    pre_pad = windowsize ÷ 2
    post_pad = (windowsize - 1) ÷ 2
    if nrow(pd) < windowsize
        return DataFrame(date=Date[], total=Float64[], std=Float64[], ci_low=Float64[], ci_high=Float64[])
    end

    idx = (pre_pad + 1):(nrow(pd) - post_pad)
    means = Float64[]
    stds = Float64[]
    ci_lows = Float64[]
    ci_highs = Float64[]

    for i in idx
        w = pd.num[(i - pre_pad):(i + post_pad)]
        m = mean(w)
        s = std(w)
        ci = 1.96 * s / sqrt(windowsize)
        push!(means, m)
        push!(stds, s)
        push!(ci_lows, m - ci)
        push!(ci_highs, m + ci)
    end

    return DataFrame(
        date=pd.date[idx],
        total=means,
        std=stds,
        ci_low=ci_lows,
        ci_high=ci_highs,
    )
end

function get_activity(comments, submissions, topic, subreddit; windowsize = 14, start_date=Date(2020, 1, 1), end_date=Date(2022, 12, 31))
    selector=row->row.topic==topic && row.subreddit==subreddit
    comment_act =  rolling_per_day(comments; selector, windowsize)
    submission_act = rolling_per_day(submissions; selector, windowsize)
    activity = DataFrame(date=start_date:Day(1):end_date)
    leftjoin!(activity, comment_act, on=:date, makeunique=true)
    leftjoin!(activity, submission_act, on=:date, makeunique=true)
    rename!(activity,
        :total => :nb_coms,
        :std => :nb_coms_std,
        :ci_low => :nb_coms_ci_low,
        :ci_high => :nb_coms_ci_high,
        :total_1 => :nb_subs,
        :std_1 => :nb_subs_std,
        :ci_low_1 => :nb_subs_ci_low,
        :ci_high_1 => :nb_subs_ci_high,
    )
    activity.nb_posts = coalesce.(activity.nb_coms, 0.) .+ coalesce.(activity.nb_subs, 0.)
    activity.nb_posts_std = sqrt.(coalesce.(activity.nb_coms_std, 0.).^2 .+ coalesce.(activity.nb_subs_std, 0.).^2)
    ci_delta = 1.96 .* activity.nb_posts_std ./ sqrt(windowsize)
    activity.nb_posts_ci_low = activity.nb_posts .- ci_delta
    activity.nb_posts_ci_high = activity.nb_posts .+ ci_delta
    return activity
end

function build_linear_model(data, knots)
    t = Float64.([(d - data.date[1]).value for d in data.date])
    n = length(t)

    cols = [ones(n), t]
    for k in knots
        t_k = Float64((k - data.date[1]).value)
        D = Float64.(data.date .>= k)
        push!(cols, D)
        push!(cols, D .* (t .- t_k))
    end
    X = hcat(cols...)

    model = lm(X, data.nb_posts)
    return model, X
end

function get_trend(data, knots; start_date=Date(2020, 1, 1), end_date=Date(2022, 12, 31))
    model, X = build_linear_model(data, knots)
    trend = DataFrame(date=data.date, vals=X * coef(model))
    drange = max(start_date, minimum(data.date)):Day(1):min(end_date, maximum(data.date))
    return trend[ein(trend.date, drange), :]
end

function get_its_results(data, knots)
    model, _ = build_linear_model(data, knots)
    β = coef(model)
    CI = confint(model)

    results = DataFrame(
        event_idx=Int[],
        event_date=Date[],
        baseline_change=Float64[],
        baseline_change_ci_low=Float64[],
        baseline_change_ci_high=Float64[],
        trend_change=Float64[],
        trend_change_ci_low=Float64[],
        trend_change_ci_high=Float64[],
    )

    for (i, k) in enumerate(knots)
        lvl_idx = 2 * i + 1
        tr_idx = 2 * i + 2

        lvl = β[lvl_idx]
        tr = β[tr_idx]

        lvl_CI_l, lvl_CI_u = CI[lvl_idx, :]
        tr_CI_l, tr_CI_u = CI[tr_idx, :]

        push!(results, (i, k, lvl, lvl_CI_l, lvl_CI_u, tr, tr_CI_l, tr_CI_u))
    end

    return results
end

function format_effect_with_ci(effect, ci_low, ci_high)
    effect_str = @sprintf("%.2f", effect)
    if !(ci_low <= 0 <= ci_high)
        effect_str = "\\textbf{$effect_str}"
    end
    return "$(effect_str) & [$( @sprintf("%.2f", ci_low)), $( @sprintf("%.2f", ci_high))]"
end

function write_its_table_latex(its_by_topic, output_path)
    open(output_path, "w") do io
        println(io, "\\begin{tabular}{p{.25\\textwidth}lcccc}\\hline")
        println(io, "\\textbf{Event} & \\textbf{Date} & \\(\\Delta\\beta_0\\) & CI 95\\% & \\(\\Delta\\beta_1\\) & CI 95\\% \\\\\\hline")

        for topic in topics
            haskey(its_by_topic, topic) || continue
            df = sort(its_by_topic[topic], :event_date)
            labels = get(KEYDATES, topic, DataFrame(:label => String[])).label
            topic_title = get(TOPIC_TITLES, topic, string(topic))

            println(io, "\\emph{$topic_title} &&&&&\\\\")
            for i in 1:nrow(df)
                event_label = i <= length(labels) ? labels[i] : "Event $i"
                level_txt = format_effect_with_ci(df.baseline_change[i], df.baseline_change_ci_low[i], df.baseline_change_ci_high[i])
                trend_txt = format_effect_with_ci(df.trend_change[i], df.trend_change_ci_low[i], df.trend_change_ci_high[i])

                println(io, "$(event_label) & $(df.event_date[i]) & $(level_txt) & $(trend_txt) \\\\")
            end
            println(io, "\\hline")
        end

        println(io, "\\end{tabular}")
    end
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

function get_title(submissions, submission_id)
    idx = findfirst(isequal(submission_id), submissions.id)
    return idx !== nothing ? submissions.title[idx] : ""
end

function get_negative_posts(comments, submissions, topic, subreddit, date; language="en", sentiment_col=:bert, nb=10)
    selection = comments[comments.topic .== topic .&& comments.subreddit .== subreddit .&&
        select_lang(comments, language) .&& comments.date .== date,:]
    valid = remove_invalid_rows(selection, [sentiment_col, ])
    valid.w_sent = valid.score .* valid[!, sentiment_col]
    negative_posts = DataFrame(
        date=Date[], submission_id=String[], comment_id=String[], title=String[],
        body=String[], score=Float64[], sent=Float64[], w_sent=Float64[]
    )
    sort!(valid, :w_sent)
    for i in 1:nb
        push!(negative_posts, Tuple(valid[i, [:date, :submission_id, :id, :author, :body, :score, sentiment_col, :w_sent]]))
        negative_posts[end, :title] = get_title(submissions, negative_posts[end, :submission_id])
    end
    return negative_posts
end

isdir(path) || mkdir(path)
isdir(joinpath(path, "negative-days")) || mkdir(joinpath(path, "negative-days"))

for subreddit in subreddits
    its_by_topic = Dict{Topic, DataFrame}()
    for topic in topics
        pd = get_activity(comments, submissions, topic, subreddit; start_date, end_date, windowsize)
        CSV.write(joinpath(path, "$(subreddit)_$(topic).csv"), pd[!,[:date, :nb_posts, :nb_posts_std, :nb_posts_ci_low, :nb_posts_ci_high]])
        trend = get_trend(pd, KEYDATES[topic].date; start_date, end_date)
        CSV.write(joinpath(path, "$(subreddit)_$(topic)_trend.csv"), trend)
        its_results = get_its_results(pd, KEYDATES[topic].date)
        its_by_topic[topic] = its_results
        CSV.write(joinpath(path, "$(subreddit)_$(topic)_its.csv"), its_results)
        @info "$(subreddit) $(topic) maximum: $(maximum(pd.nb_posts)) at $(pd.date[argmax(pd.nb_posts)]) ($(maximum(pd.nb_posts) / mean(pd.nb_posts)) x mean)"
    end
    write_its_table_latex(its_by_topic, joinpath(path, "$(subreddit)_its_table.tex"))
end

open(joinpath(path, "negative-days", "00_summary.csv"), "w") do f
    for subreddit in subreddits
        for topic in topics
            neg_days = identify_negative_days(comments, topic, subreddit; q=negative_day_quantile, threshold=negative_day_thres)
            println(f, "$subreddit,$(topic): "*join(neg_days, ", "))
            for date in neg_days
                neg_posts = get_negative_posts(comments, submissions, topic, subreddit, date)
                CSV.write(joinpath(path, "negative-days", "$subreddit-$(topic)-$(date).csv"), neg_posts)
            end
        end
    end
end
