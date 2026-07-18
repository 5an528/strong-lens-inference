# Presentation Speech — Strong Lens Inference

**Total: 12 minutes — 3 speakers × 4 minutes.**
Matched slide-by-slide to `StrongLensInference.pptx` (12 slides → each speaker covers 4 slides, ~1 minute per slide).
Spoken pace ≈ 135 words/min, so each part is ~520–540 words. Practice with a timer; the ⏱ marks tell you where you should be.

All numbers below match the final **50,000-sample** runs (see `RESULTS.md`, Runs 6–7, and `report/report.pdf`).

Suggested split (swap freely):

| Speaker | Slides | Theme |
|---|---|---|
| Speaker 1 — *Sayed Atique Newaz* | 1–4 | The problem: lensing physics & why SBI |
| Speaker 2 — *Noureen Alam Meem* | 5–8 | The method: simulation, training, core results |
| Speaker 3 — *Mashuk Khan* | 9–12 | The validation: physics tests & conclusions |

---

## SPEAKER 1 — The problem (0:00 – 4:00)

### Slide 1 — Title ⏱ 0:00

Good morning everyone. We are — myself, Noureen, and Mashuk — and today we're going to show you how we taught a neural network to do gravitational lens modelling: to look at a single noisy telescope image and tell us the physical parameters of the galaxy that created it — not just point estimates, but full probability distributions, in milliseconds.

The background image here is real: it's the galaxy cluster Abell 370, taken by Hubble. Every stretched arc you can see in it is a distant galaxy, distorted by gravity. That distortion is our data.

### Slide 2 — Gravity's magnifying glass ⏱ 0:45

So what is strong gravitational lensing? General relativity tells us that mass bends light. When a massive foreground galaxy sits almost exactly between us and a distant background galaxy, the light from that background galaxy bends around it on its way to us.

The alignment decides what we see. A small offset produces multiple images and stretched arcs. A near-perfect alignment produces something spectacular — a complete circle of light called an Einstein ring. You can see one on the right: the Cosmic Horseshoe, where a blue background galaxy has been smeared into an almost closed ring.

And here's the key point: this distortion is not random. The radius of the ring tells you the mass enclosed by the lens. The shape of the arcs tells you how elliptical the lens is and what tidal forces act on it. The image is a fingerprint of the lens. Our job is to read that fingerprint backwards — the inverse problem.

### Slide 3 — Real examples ⏱ 1:50

And nature runs this experiment for us all the time. These are three real Hubble observations. On the left, the Cosmic Horseshoe again — a near-complete ring. In the middle, a galaxy group that lenses background galaxies into arcs that famously look like a smiley face. And on the right, a supernova — a single exploding star — that appears four times in one image, because the foreground galaxy splits its light into four separate paths.

These are exactly the kinds of morphologies — rings, arcs, multiple images — that our simulated data has to cover.

### Slide 4 — Why simulation-based inference ⏱ 2:35

So why is this hard? Simulating a lens forward is easy — given the parameters, the lenstronomy package produces the image. But going backwards is not. Between PSF blur, pixel noise, and parameter degeneracies, the likelihood — the probability of the image given the parameters — has no tractable form. Classical methods like MCMC work around this, but they fit one system at a time and take hours per lens.

Simulation-based inference flips the problem: if you can simulate, you can learn. We use BayesFlow to train a neural network on tens of thousands of simulated image–parameter pairs. Training happens once. After that, the network gives us the full posterior distribution for *any* new image in milliseconds. That's called amortized inference.

The table on the right shows what we're inferring: the Einstein radius, two ellipticity components, two external shear components, the source position and size — eight parameters, or nine when we include the external convergence kappa, which will matter later in Mashuk's part.

⏱ 3:50 — **Handoff:** "So that's the problem. Noureen will now show you how we actually built and trained this."

---

## SPEAKER 2 — The method and core results (4:00 – 8:00)

### Slide 5 — Simulating realistic lenses ⏱ 4:00

Thank you. Since the network learns entirely from simulations, the simulations have to be realistic — the network can only be as good as the data we feed it.

Our lens model is a singular isothermal ellipsoid — the standard model for galaxy-scale lenses — plus external shear, and optionally that convergence sheet kappa. The background source is an elliptical Sérsic profile, ray-traced through the lens equation with lenstronomy. Then we make it look like a real observation: 64 by 64 pixels, a Gaussian point-spread function, and two noise sources — Gaussian background noise and Poisson shot noise, whose strength grows with the signal, just like a real CCD.

One thing we learned the hard way: our first datasets were too faint, with a peak signal-to-noise around 4. We ran a tuning sweep on the source brightness and brought the median SNR to about 16, inside the 10-to-30 target range. You'll see in a moment how dramatically that mattered.

On the right you see one simulated system — the clean lensed arcs on the left, and the noisy image the network actually sees on the right.

### Slide 6 — Training ⏱ 5:10

We generated **fifty thousand** training images — the upper end of the assignment's simulation budget — plus two thousand for validation and three hundred held-out test systems. Because the simulations are independent, generation parallelizes almost perfectly: with fifteen CPU workers, the whole dataset takes about a minute.

The architecture has two parts. A small convolutional network compresses each image into a 48-dimensional summary — it learns by itself which image features matter. Then a conditional coupling flow, a type of normalizing flow, maps that summary to the full joint posterior over all parameters.

Training took about two hours on a single RTX 3060 GPU — the GPU's four-to-five-times speedup over CPU is what made a dataset this size practical at all. We also armed early stopping as a safeguard, but it never fired: as the loss curve shows, the validation loss decreased through all forty epochs and stayed *below* the training loss the whole time. No overfitting — our earlier eight-thousand-sample pilot did overfit, and scaling the data fixed it.

### Slide 7 — Parameter recovery ⏱ 6:10

So — does it work? These panels show, for the three hundred test systems the network never saw, the inferred value against the true value; the error bars are the posterior spread, and each panel quotes the correlation between estimate and truth. Perfect recovery would put every point on the diagonal.

The Einstein radius, the source position, and the source size are recovered with correlations of 0.999 — R-squared of one point zero zero. Ellipticity: R-squared 0.94 and 0.92. External shear is the hardest — 0.86 and 0.80 — and that's physically reasonable, because shear produces only subtle image distortions.

And remember the SNR fix I mentioned? At the original low SNR, shear recovery was 0.21 and 0.06 — essentially nothing. Same pipeline, same network. Image quality is everything.

### Slide 8 — Are the error bars honest? ⏱ 7:10

But in Bayesian inference, being accurate isn't enough — the *uncertainties* have to be trustworthy. We test this with simulation-based calibration: if the posteriors are honest, the true value should rank uniformly within its own posterior samples. These panels show the rank distribution as a difference from perfect uniformity, with a ninety-five percent confidence band.

Ellipticity and shear: inside the band — calibrated. The four best-measured parameters trace an S-shape outside the band: their error bars are slightly too *wide*. That's the safe direction to be wrong in — a method that slightly underclaims is far better than an overconfident one. And here's the diagnostic punchline: this pattern is identical at twenty and fifty thousand training samples, so it's a limit of the network's capacity, not of the data — we know exactly which knob to turn next.

⏱ 7:50 — **Handoff:** "So we can recover the parameters, and we can trust the error bars. Mashuk will now push the model harder — with a test where the right answer is 'I don't know'."

---

## SPEAKER 3 — Validation and conclusions (8:00 – 12:00)

### Slide 9 — Posterior predictive ⏱ 8:00

Thank you. One more consistency check before the hard test. If the posterior is right, then parameters drawn from it, pushed back through the simulator, should reproduce the observation. That's a posterior predictive check.

On the far left is the noisy observed image. The four panels next to it are independent posterior draws, re-simulated. The ring radius, the bright knots, and the light distribution around the ring are all reproduced. The draws differ mainly in the exact brightness of the knots — which is precisely the part the noise genuinely hides. The network reconstructs what's knowable, and stays appropriately uncertain about what isn't.

### Slide 10 — The mass-sheet degeneracy ⏱ 9:00

Now the most interesting result. Lensing theory contains a famous blind spot called the mass-sheet degeneracy: if you add a uniform sheet of mass — the convergence kappa — and rescale the source, the image barely changes at all. Which means kappa fundamentally *cannot* be measured from a single image. No method should be able to do it.

So we tested our network on exactly that. We added kappa as a ninth parameter and retrained on a fresh fifty thousand simulations. What should a *correct* method do? It should fail — honestly.

And it does. The posterior for kappa stays essentially as wide as the prior — contraction of only sixteen percent — the network learned almost nothing about it, exactly as theory demands. But look at the correlation: minus 0.96 between kappa and the Einstein radius. The network didn't just give up on kappa — it discovered the precise trade-off the lens equations predict, and its posterior collapses onto that degeneracy ridge. In fact, that correlation *strengthened* as our data got better — minus 0.83 in the noisy pilot, minus 0.96 now — because a sharper posterior hugs the ridge more tightly. And the collateral damage is surgical: only the Einstein radius drops, from 1.00 to 0.94. Every other parameter is untouched.

Nobody told the network about this degeneracy. It rediscovered a piece of gravitational lensing theory purely from simulated data. For us, this is the strongest evidence that the posteriors mean something physical.

### Slide 11 — Conclusions ⏱ 10:30

To conclude. We built an end-to-end simulation-based inference pipeline: simulate, train once, then get full posteriors in milliseconds per lens. At realistic signal-to-noise, recovery is essentially perfect for the Einstein radius and source properties, and good — 0.80 to 0.94 — for ellipticity and shear. The uncertainties are calibrated or mildly conservative, and known physical degeneracies emerge exactly where theory says they must.

And the milliseconds matter: upcoming surveys like Euclid and Rubin-LSST are expected to find on the order of a hundred thousand new lenses. At hours per lens, classical fitting simply cannot keep up. Amortized inference is built for exactly that future. Next steps for us: a slightly larger network — we showed the remaining conservatism is capacity-limited, not data-limited — more complex sources, and real survey images.

### Slide 12 — Thank you ⏱ 11:30

That's our project. The universe on this slide seems to approve — this smiley face is a real Hubble image of a gravitational lens. Thank you for listening, and we're happy to take questions.

---

# Prepared Q&A

Assign by topic: physics questions → Speaker 1, method/training → Speaker 2, results/validation → Speaker 3. Whoever answers, keep it under ~30 seconds.

**Q1. Why not just use MCMC? (→ Speaker 1 or 2)**
MCMC needs an explicit likelihood and fits one system at a time — hours per lens. Our network trains once (~2 h) and then handles any number of lenses in milliseconds each. For one lens MCMC is fine; for the ~100,000 lenses Euclid and LSST will find, amortized inference is the only approach that scales. Also, MCMC's answer is only as good as the likelihood you write down; we never need to write one.

**Q2. How do you know the network's posterior is actually correct? (→ Speaker 3)**
Four independent checks: (1) recovery — posterior medians match truth on 300 unseen systems; (2) simulation-based calibration — rank ECDFs against a 95% band, which found and honestly quantified a mild conservatism; (3) posterior contraction and z-scores — strong information gain with no bias; (4) posterior predictive — re-simulated draws reproduce the observation. And the strongest check: the mass-sheet degeneracy appears exactly where theory demands.

**Q3. What exactly is the mass-sheet degeneracy? (→ Speaker 3)**
A transformation of the lens model: scale the mass distribution down by (1−κ) and add a uniform mass sheet κ, while rescaling the source — the observed image is (nearly) unchanged. So single-image data cannot distinguish these models. It's broken only with extra information: time delays, stellar kinematics, or absolute source sizes/magnifications.

**Q4. Would this work on a real telescope image? (→ Speaker 2)**
Not directly yet — that's the honest answer. The network is only as good as its simulator, and real data has complications we didn't simulate: realistic PSFs, cosmic rays, the lens galaxy's own light, more complex source structure. The pipeline is built for that upgrade — you improve the simulator and retrain; the inference machinery stays identical. That's our main next step.

**Q5. Why is the shear (γ2 = 0.80) still the worst parameter? (→ Speaker 3)**
Two reasons. Physically, external shear produces very subtle image distortions, and our prior range is small (0 to 0.08) — there just isn't much signal in the pixels. Statistically, when data is weakly informative, the posterior stays broad and its mean is pulled toward the prior mean, lowering R². Note the trajectory though: 0.06 at low SNR, 0.65 at 20k samples, 0.80 at 50k — and the calibration plots show the shear posteriors are *honest* at every stage: wide, but correct.

**Q6. Why a Sérsic source? What if the real source is clumpier? (→ Speaker 2)**
Sérsic is the standard parametric model for galaxy light and keeps θ low-dimensional. A clumpy real source is model misspecification — the network would be confidently wrong in ways calibration on simulations can't catch. Fixes: more flexible source models (e.g., shapelets or a generative model trained on real galaxy images) in the simulator, then retrain.

**Q7. How is your SNR defined, and why did it matter so much? (→ Speaker 2)**
Peak SNR: brightest lensed pixel over the local noise level (background plus Poisson). Our first datasets had median ~4 — arcs barely above noise, so the network could only learn the strongest features and shear was hopeless (R² 0.06). After a brightness sweep we hit median ~16, inside the 10–30 spec, and every parameter improved dramatically. It shows the information is in the image or it isn't — no network can invent it.

**Q8. What's the difference between ellipticity (e1, e2) and shear (γ1, γ2)? (→ Speaker 1)**
Ellipticity is *internal* — the shape of the lens galaxy's own mass distribution. Shear is *external* — the tidal gravitational field from other structures near the line of sight (neighboring galaxies, the group environment). Both stretch the image, which is partly why they're harder to separate than, say, the Einstein radius.

**Q9. How long does the whole pipeline take? (→ Speaker 2)**
Dataset generation: about one minute for 52,000 images, parallelized over 15 CPU workers — lenstronomy simulation is cheap per image. Training: ~2 hours per model on one RTX 3060 (the GPU gave a 4–5× per-step speedup over CPU; that's what made 50k samples feasible). Inference afterwards: 2,000 posterior samples for all 300 test systems in ~0.3 seconds — milliseconds per lens.

**Q10. Could you constrain κ at all? (→ Speaker 3)**
Not from a single image — that's a theorem, not a limitation of our method. Our measured contraction of the kappa posterior versus the prior was only +0.16, i.e., nearly nothing learned, which is the correct answer. To actually measure kappa you need to break the degeneracy with external data: time-delay measurements of a lensed variable source, velocity dispersion of the lens galaxy, or independent knowledge of the source.

**Q11. What is BayesFlow doing that a normal CNN regression wouldn't? (→ Speaker 2)**
A CNN regression gives one number per parameter and no uncertainty. BayesFlow's coupling flow learns the *full joint posterior distribution* — widths, skews, and crucially correlations between parameters. The kappa–θE correlation of −0.96 is exactly the kind of structure a point-estimate regressor could never express.

**Q12. Your calibration isn't perfect — isn't that a problem? (→ Speaker 3, honest answer)**
It's a mild, quantified conservatism for the four best-measured parameters: their credible intervals are slightly too wide, never too narrow, so no downstream analysis is misled. More importantly, we isolated its cause: the S-shape is identical at 20k and 50k training samples, so it's network capacity, not data. The fix is known (larger summary dimension or deeper flow) — we report it as a limitation with a diagnosis, which is exactly what these calibration tools are for.

**Q13. Did you validate against a classical method on the same data? (→ Speaker 3, honest answer)**
Not yet — a per-system MCMC fit with lenstronomy on a handful of test lenses would be the natural benchmark, and it's on our list. What we do have is ground truth from simulations, which is arguably a stronger test than method-vs-method agreement, plus calibration checks MCMC itself doesn't automatically give you.

---

## Timing rescue plan

If you're running over, cut in this order (least damage first):
1. Speaker 1, Slide 3 — compress the three examples to one sentence ("rings, arcs, multiple images — all real Hubble data").
2. Speaker 2, Slide 5 — skip the instrument details (pixels/PSF), keep the SNR story.
3. Speaker 3, Slide 9 — one sentence: "posterior draws re-simulated look like the data."

Never cut: Slide 4 (why SBI), Slide 7 (recovery numbers), Slide 10 (degeneracy — it's the punchline).
