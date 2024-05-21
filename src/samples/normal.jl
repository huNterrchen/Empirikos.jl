abstract type AbstractNormalSample{T} <: ContinuousEBayesSample{T} end

"""
    NormalSample(Z,σ)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z \\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown.
The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> NormalSample(0.5, 1.0)          #Z=0.5, σ=1
N(0.5; μ, σ=1.0)
```
"""
struct NormalSample{T,S} <: AbstractNormalSample{T}
    Z::T
    σ::S
end


function NormalSample(σ::S) where {S}
    NormalSample(missing, σ)
end



"""
    StandardNormalSample(Z)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 =1``.

```math
Z \\sim \\mathcal{N}(\\mu, 1)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> StandardNormalSample(0.5)          #Z=0.5
N(0.5; μ, σ=1.0)
```
"""
struct StandardNormalSample{T} <: AbstractNormalSample{T}
    Z::T
end

StandardNormalSample() = StandardNormalSample(missing)

eltype(Z::AbstractNormalSample{T}) where {T} = T
support(Z::AbstractNormalSample) = RealInterval(-Inf, +Inf)

response(Z::AbstractNormalSample) = Z.Z
var(Z::AbstractNormalSample) = Z.σ^2
var(Z::StandardNormalSample) = one(eltype(response(Z)))
var(Z::StandardNormalSample{Missing}) = 1.0

std(Z::AbstractNormalSample) = Z.σ
std(Z::StandardNormalSample) = one(eltype(response(Z)))
std(Z::StandardNormalSample{Missing}) = 1.0

nuisance_parameter(Z::AbstractNormalSample) = std(Z)
primary_parameter(::AbstractNormalSample) = :μ

likelihood_distribution(Z::AbstractNormalSample, μ) = Normal(μ, std(Z))


function Base.show(io::IO, Z::AbstractNormalSample)
    Zz = response(Z)
    print(io, "N(", Zz, "; μ, σ=", std(Z),")")
end





# Targets

# TODO: Note this is not correct for intervals.
function cf(target::MarginalDensity{<:AbstractNormalSample}, t)
    error_dbn = likelihood_distribution(location(target))
    cf(error_dbn, t)
end


# Conjugate computations
function default_target_computation(::BasicPosteriorTarget,
    ::AbstractNormalSample,
    ::Normal
)
    Conjugate()
end

function marginalize(Z::AbstractNormalSample, prior::Normal)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end


function posterior(Z::AbstractNormalSample, prior::Normal)
    z = response(Z)
    sigma_squared = var(Z)
    prior_mu = mean(prior)
    prior_A = var(prior)

    post_mean =
        (prior_A) / (prior_A + sigma_squared) * z +
        sigma_squared / (prior_A + sigma_squared) * prior_mu
    post_var = prior_A * sigma_squared / (prior_A + sigma_squared)
    Normal(post_mean, sqrt(post_var))
end



# Uniform-Normal

struct UniformNormal{T} <: Distributions.ContinuousUnivariateDistribution
    a::T 
    b::T
    σ::T
end

Distributions.@distr_support UniformNormal -Inf Inf

function Distributions.pdf(d::UniformNormal, x::Real)
    base_normal = Normal(0.0, d.σ)
    a = d.a 
    b = d.b 
    (cdf(base_normal, b-x) - cdf(base_normal, a-x)) / (b-a)
end

function Distributions.cdf(d::UniformNormal, x::Real)
    σ = d.σ
    a = d.a 
    b = d.b 
    right_limit = (x-a)/σ
    left_limit = (x-b)/σ
    improper_integral(u) = u*cdf(Normal(), u) + pdf(Normal(), u)
    σ*(improper_integral(right_limit) - improper_integral(left_limit))/(b-a)
end


function marginalize(Z::AbstractNormalSample, prior::Uniform)
    UniformNormal(prior.a, prior.b, std(Z))
end

# Target specifics
function Base.extrema(density::MarginalDensity{<:AbstractNormalSample{<:Real}})
    (0.0, 1 / sqrt(2π * var(location(density))))
end
