using PowerLaws

include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO]))

subreddits = ["belgium"]
topics = [lockdown, mask, vaccin]
langs = ["en", "nl", "fr", "de"]
categories = Dict([:topic=>topics,:language=>langs])

path = joinpath(RESULT_DIR, "01_degree_data")

isdir(path) || mkdir(path)

for subreddit in subreddits
    sub_coms = comments[comments.subreddit .== subreddit, :]
    sub_subs = submissions[submissions.subreddit .== subreddit, :]

    users = get_users(sub_coms, sub_subs, categories)
    open(joinpath(RESULT_DIR, "01_degree_data", "$(subreddit)_number_table.tex"), "w") do io
        println(io, "\\begin{tabular}{r|ccc}")
        println(io, "  & Users & Comments & Submissions \\\\\\hline")
        println(io, "  \\textbf{Total} & $(length(users.author)-1) & $(sum(users.nc)) & $(sum(users.ns)) \\\\\\hline")
        println(io, "  \\textbf{Per language} & & & \\\\")
        for lang in langs
            lang_users = users[users[!,"nt_$(lang)"] .> 0, :]
            lang_coms = sum(lang_users[!, "nc_$lang"])
            lang_subs = sum(lang_users[!, "ns_$lang"])
            println(io, "  $lang & $(nrow(lang_users)-1) & $(lang_coms) & $(lang_subs) \\\\")
        end
        println(io, "  \\textbf{Per topic} & & & \\\\")
        for topic in topics
            topic_users = users[users[!,"nt_$(topic)"] .> 0, :]
            topic_coms = sum(topic_users[!, "nc_$topic"])
            topic_subs = sum(topic_users[!, "ns_$topic"])
            println(io, "  $topic & $(nrow(topic_users)-1) & $(topic_coms) & $(topic_subs) \\\\")
        end
        println(io, "\\end{tabular}")
    end

    # Without deleted
    users = users[users.author .!= AUTHOR_DELETE, :]

    fit = DataFrame(topic=String[], alpha=Float64[], minx=Float64[], KS=Float64[])

    for topic in topics
        col = "nt_$(topic)"
        ds = sort(setdiff(unique(users[!, col]), [0]))
        counts = [count(x->x==d, users[!, col]) for d in ds]
        df = DataFrame(:ds=>ds, :counts=>counts)
        for xmins in [[1,], 1:10]
            params, KS = estimate_parameters(users[users[!, col] .> 0, col], DiscretePowerLaw, xmins=xmins)
            α = params.α
            A = df.counts[10] / (df.ds[10]^(-1. *α))
            df.fit = A*(df.ds.^(-1. * α))
            CSV.write(joinpath(path, "$(subreddit)_$(topic)_$(params.θ).csv"), df)
            push!(fit, (string(topic), α, params.θ, KS))
        end
    end

    CSV.write(joinpath(path, "$(subreddit)_fit_info.csv"), fit)
end

