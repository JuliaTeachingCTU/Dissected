# # Backpropagation through the void: optimizing control variates for black-box gradient estimation
#
# The paper tries to solve the problem of estimating the gradient of an expectation with respect
# to a discrete parameterized distribution. An interesting feature of the method is that 
# the estimator is unbiassed, but it uses relaxed samples provided by reparametrization trick
# as a control variate reducing the variance. Moreover, unlike other methods parameters of the relaxation, importantly the temperature are optimized.

using Flux
using Flux.Zygote
using Distributions
using Statistics
using GLMakie

# Let's start with an easy problem. Let's assume we have a function `f` of a bernoulli variable. 
# In the paper, they use
f(b) = (b - 0.45)^2
# We can compute an expectation of `f` with respect to Bernoulli random variable
# with probability `p(b=1|θ)` is given as 
function expectation(f, θ)
	f(1) * θ + f(0) * (1 - θ)
end
# Note that in reality, the support of the discrete random variable will be so large 
# that this exact computation will be impossible to compute exactly and we have to resort to 
# monte carlo estimation. For example the paper discusses a case of 200 hundreds Bernoulli variables,
# which has the support of $2^200$.  The monte-carlo estimate can be computed as 
function monte_carlo(f, θ, n)
	mean(f.(rand(Bernoulli(θ), n)))
end
# where `n` is the number of samples. The estimator is unbiassed, but it has certain variance, which
# can be seen as follows.
lines(20, [std(monte_carlo(f, 0.3, 2^n) for _ in 1:1000) for n in 1:20])

# But we are interested in gradients. The classic method to compute stochastic estimate of 
# the gradient of parameters of the discrete distribution is REINFORCE. The method is based on the
# fact that the gradient of the expectation of the function can be computed as 

logpdf_bernoulli(θ, x) = x * log(θ) + (1-x)*log(1-θ)

function reinforce(f, θ, n) 
	mean(1:n) do i 
		x = rand(Bernoulli(θ))
		f(x) * only(Zygote.gradient(θ -> logpdf_bernoulli(θ, x), θ))
	end
end

lines(1:20, [std(reinforce(f, 0.3, 2^n) for _ in 1:1000) for n in 1:20])

# First, we verify that the estimate of REINFORCE method is by comparing it to the true value. 
reinforce(f, θ, 100000) 
only(Zygote.gradient(θ -> expectation(f, θ), θ))

# The REINFORCE method is unbiassed, but it is known to have high variance. The reinforce trick can 
# be of course computed exactly as
function exact_reinforce(f, θ) 
	mapreduce(+, (1f0,0f0)) do x
		pdf(Bernoulli(θ), x) * f(x) * Zygote.gradient(θ -> logpdf_bernoulli(θ, x), θ)[1]
	end
end

exact_reinforce(f, θ)  ≈ only(Zygote.gradient(θ -> expectation(f, θ), θ))

# which of course works. Finally, we can measure and plot the variance of reinforce estimator as
lines(1:20, [std(reinforce(f, 0.3, 2^n) for _ in 1:1000) for n in 1:20])

# which would serve as a baseline on which we want to improve.



# If the function `f` accepts real values, a popular low-variance estimator of the expectation is based on 
# reparametrization trick.The idea is to introduce noise variable `u` from some known distribution with fixed
# parameters (in our case uniform $U(0,1)$) and through differentiatble transformation dependent on parameters 
# of interest convert it to the desired distribution. Because the transformation needs to be differentiable,
# this trick produces continuous (not discrete) distributions. But if `f` accepts real values, we get some 
# estimate of something, which frequently works.

# For our case of Bernoulli, we first define define random variable `p(z|θ)`, for which it holds that 
# `heaviside(p(z|θ))` is Bernoulli distributed with probability `θ`. We can define it as 
logit(x::T) where {T<:Real} = log(x / (one(T)-x)) 
pz_θ(θ::Real, u::Real) = logit(θ) - logit(u)
pz_θ(θ::T) where {T<:Real} = logit(θ) - (Zygote.@ignore logit(rand(T)))
heaviside(z) = z >= 0

# The correctness can be be verified by showing that `mean((heaviside ∘ pz_θ)(0.3f0) for _ in 1:100000) = 0.29952`.
# The smooth approximation of the heaviside is the sigmoid function. The sharpness of the approximation 
# is controlled by the parameter `temperature`. As the `temperature` goes to zero (in logspace to `-∞`), 
# the approximation is closer to Bernoulli.
bernoulli_softmax(z, log_temperature) = Flux.σ(z / exp(log_temperature))

# Below plot shows the histogram of the smooth approximation for different temperature. 
fig, ax,_ = hist([(heaviside ∘ pz_θ)(0.3f0) for _ in 1:10000], bins = 100, label = "Bernoulli", normalization = :pdf);
for τ in -3:0
	hist!(ax, [(bernoulli_softmax(pz_θ(0.3f0), τ)) for _ in 1:10000], bins = 100, label = "Soft bernouli with $(τ)", normalization = :pdf)
end
fig

# As said above, the smooth approximation is differentiable with respect to `θ` and we can therefore
# directly estimate the gradient of the expectation using stochastice monte-carlo.
function reparam(f, θ, n, τ) 
    mean(1:n) do i 
        only(Zygote.gradient(θ -> f(bernoulli_softmax(pz_θ(θ), τ)), θ))
    end
end

# This estimator has lower variance than the reinforce but it is biassed. Below figure shows on the left 
# mean of estimates and on the right their variance.
fig = Figure(size = (800, 1000))
ga = fig[1, 1] = GridLayout()
axleft = Axis(ga[1, 1])
axright = Axis(ga[1, 2])

nmax = 15
meanstd(vals) = (mean(vals), std(vals))
stats = [meanstd([reinforce(f, 0.3, 2^n) for _ in 1:1000]) for n in 1:nmax]
lines!(axleft, 1:nmax, first.(stats), label = "Bernoulli - reinforce");
lines!(axright, 1:nmax, last.(stats), label = "Bernoulli - reinforce");
for τ in -3:0
    stats = [meanstd([reparam(f, 0.3, 2^n, τ) for _ in 1:1000]) for n in 1:nmax]
    lines!(axleft, 1:nmax, first.(stats) , label = "Soft bernouli with $(τ)")
    lines!(axright, 1:nmax, last.(stats), label = "Soft bernouli with $(τ)")
end
axislegend(axright)
fig
# What can see that the estimate of the reparametrized Bernoulli is biassed and the biass decreases with 
# the temperature. Contrary, the variance of the estimate increases with the temperature. Finally we 
# see that the variance is higher than that of Reinforce method, but that can be caused by the fact that we 
# have only one Bernoulli variable.

# So far, we have two approaches. One, which is biassed and has low variance and the other, 
# which is unbiassed but it has high variance.
# The ideal introduced by Rebar (Rebar: Low-variance, unbiased gradient estimates for discrete latent variable models),
# is to combine the two methods. The idea is to use the reparametrization trick to create a control variate
# for the reinforce method and optimize its parameters to have low variance.
# 
# The idea is followig. We have a function `f` and a Bernoulli random variable with parameter `θ`.
# Reinforce computes the gradient of the expectation of `f` with respect to `θ` as 
# ``\mathbb{E}_{b \sim p(b|\theta)}\left[f(b)\logp(b|\theta) \right]``
# and the reparametrization trick as 
# ``\mathbb{E}_{u \sim U[0,1]}f(\phi(\theta,u))\right]``
# where ``\phi(\theta,u)` is the above reparametrization trick implemented by composition `bernoulli_softmax ∘ pz_θ`.
# 
# This estimator is called LAX and is defined as
# 
# ``\mathbb{E}_{b \sim p(b|\theta)}\left[f(b)\logp(b|\theta) - \mathbb{E}_{v \sim p(v|b,\theta)}\left[f(\phi(\theta,v)) \right]  \right] + \mathbb{E}_{u \sim U[0,1]}f(\phi(\theta,u))\right]``
# 
# where we need to ensure that the distribution of ``v \sim p(v|b,\theta)p(b|\theta)`` is uniform on `[0,1].`


# Below function generates `v` from a noise \sim U[0,1] and b
function conditional_noise(θ, b, noise)
    (1-b) * (noise * (1 - θ) + θ) + b * noise * θ
end

conditional_noise(θ, b) = conditional_noise(θ, b, rand())


# We can see that the distribution of both noises are matching.
hist([conditional_noise(0.3, rand(Bernoulli(0.3)), rand()) for _ in 1:10000], bins = 100, normalization = :pdf)


# TODO: We need to compute reinforce with respect to p(z|θ) and not with respect to p(b|θ), I guess.


# it has to hold that 
θ = 0.3
τ = 0
z̃ = map(1:10000) do _ 
    b = rand(Bernoulli(θ))
    v = conditional_noise(θ, b, rand())
    z̃ = bernoulli_softmax(pz_θ(θ,v), τ)
end
z = map(1:10000) do _
    bernoulli_softmax(pz_θ(θ,rand()), τ)
end

plt, ax, _ =hist(z̃, bins = 100, normalization = :pdf)
hist!(ax, z, bins = 100, normalization = :pdf)

function reinforce_control(f, θ, n, τ) 
    mean(1:n) do i 
        b = rand(Bernoulli(θ))
        v = conditional_noise(θ, 1-b, rand())
        z̃ = bernoulli_softmax(pz_θ(θ,v), τ)
        (f(b) - f(z̃)) * only(Zygote.gradient(θ -> logpdf_bernoulli(θ, b), θ))
    end
end

function lax(f, θ, τ, n)
    reinforce_control(f, θ, τ, n) + reparam(f, θ, τ, n)
end

fig = Figure(size = (800, 1000))
ga = fig[1, 1] = GridLayout()
axleft = Axis(ga[1, 1])
axright = Axis(ga[1, 2])

nmax = 15
stats = [meanstd([reinforce(f, 0.3, 2^n) for _ in 1:1000]) for n in 1:nmax];
lines!(axleft, 1:nmax, first.(stats));
lines!(axright, 1:nmax, last.(stats), label = "Bernoulli - reinforce");
for (τ,color) in zip(-3:0,[:yellow, :red, :brown, :green])
    stats = [meanstd([reparam(f, 0.3, 2^n, τ) for _ in 1:1000]) for n in 1:nmax]
    lines!(axleft, 1:nmax, first.(stats); color)
    lines!(axright, 1:nmax, last.(stats); color, label = "Soft bernouli with $(τ)")

    stats = [meanstd([lax(f, 0.3, 2^n, τ) for _ in 1:1000]) for n in 1:nmax]
    lines!(axleft, 1:nmax, first.(stats); color, linestyle = :dash)
    lines!(axright, 1:nmax, last.(stats); color, linestyle = :dash, label = "lax with $(τ)")
end
axislegend(axright)
fig


# Let's now put it together to implement relax

function logaddexp(x, y) 
    m =  max(x, y)
    log(exp(x - m) + exp(y - m)) + m
end

bernoulli_sample(logit_theta, noise) = logit(noise) < logit_theta

"""
    log Bernoulli(targets | theta), targets are 0 or 1.
"""
bernoulli_logprob(logit_theta, targets) = -logaddexp(0, -logit_theta * (targets * 2 - 1))

log_bernoulli(θ, x) = x * log(θ) + (1-x)*log(1-θ)

relaxed_bernoulli_sample(logit_theta, noise, log_temperature) = bernoulli_softmax(logistic_sample(noise, expit.(logit_theta)), log_temperature)


"""
    Computes p(u|b), where b = H(z), z = logit_theta + logit(noise), p(u) = U(0, 1)
"""
# function conditional_noise(logit_theta, samples, noise)
#     uprime = expit(-logit_theta)  # u' = 1 - theta
#     vprime = samples * (noise * (1 - uprime) + uprime) + (1 - samples) * noise * uprime
#     logit_theta + logit(vprime)
# end


############### REINFORCE ##################
"""
    function reinforce(θ, u, x)

    θ, u, x =  param, noise, func_vals
"""
function reinforce(θ, noise, x)
    samples = bernoulli_sample.(θ, noise)
    x .* Zygote.gradient(θ -> sum(bernoulli_logprob.(θ, samples)), θ)
end

############### CONCRETE ###################

function concrete(θ, log_temperature, noise, f)
    relaxed_samples = relaxed_bernoulli_sample.(θ, noise, log_temperature)
    f(relaxed_samples)
end

function relax(θ, u, v, log_temperature, surrogate, f)
    samples = bernoulli_sample.(θ, u)

    cond_noise = conditional_noise.(θ, samples, v)  # z tilde
    func_vals = f(samples)
    ∇surrogate = Flux.gradient(θ -> sum(concrete(θ, log_temperature, u, surrogate)), θ; nest = true)[1]

    # Flux.data(∇surrogate) ≈ [ -0.001180101555650044, 0.0002492344295998852]
    cond_surrogate = concrete(θ, log_temperature, cond_noise, surrogate)
    # isapprox(Flux.data(cond_surrogate), [-0.03494488 -0.03043753 -0.03903143], atol = 1e-8)

    ∇cond_surrogate = Flux.gradient(() -> sum(cond_surrogate), Params(θ); nest = true)[θ]
    # Flux.data(∇cond_surrogate) ≈ [0.005496772884193195, -0.0009223767689383597]

    (func_vals, reinforce(θ, u, func_vals .- cond_surrogate) .+ ∇surrogate .- ∇cond_surrogate)
end

"""

    est_params = (τ, ϕ)
    θ --- parameters of Bernoulli probability distribution
    ϕ --- parameters of surrogate network
"""
function relax_all(θ, ϕ, τ, surrogate, u, v, f)
    # Returns objective, gradients, and gradients of variance of gradients.
    θm = param(repeat(Flux.data(θ), inner = (1, size(u, 2))))
    func_vals, ∂f_∂θ = relax(θm, u, v, τ, surrogate, f)
    ∂var_∂ϕ = Flux.gradient(() -> sum(∂f_∂θ .^ 2) ./ size(u, 2), Params(ϕ); nest = true)
    
    for p in ϕ
        p.grad .= Flux.data(∂var_∂ϕ[p])
    end
    θ.grad .= dropdims(mean(Flux.data(∂f_∂θ), dims = 2), dims = 2)
    func_vals
end
