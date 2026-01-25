# A Few Useful Things to Know About Machine Learning (2012)
Paper: https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

## Core Ideas
- Generalization error is the only metric that ultimately matters.
- Data coverage and quality typically dominate algorithm choice.
- Every model embeds assumptions that eventually surface.

## Highlights

- **Generalization error is the only quantity that matters**  
  Training error and optimization success are poor proxies for real-world performance. Models with strong offline metrics often degrade immediately once exposed to live traffic because evaluation captured optimization, not generalization.

- **Data distribution assumptions are implicit and fragile**  
  Learning assumes train and test data come from the same distribution. Random cross-validation commonly hides temporal or cohort-based shifts that later trigger sharp performance collapse.

- **More features and more complexity increase overfitting risk**  
  High-dimensional representations make fitting noise easier, even as training error decreases. In production, additional features often improve offline metrics while amplifying leakage, feedback loops, or spurious correlations.

- **Evaluation procedures encode beliefs about the future**  
  Data splits and validation schemes reflect assumptions about how the world evolves. When evaluation does not mirror deployment conditions, performance estimates become confident but misleading.

## What Works
- Problem framing anchored to an actual decision.
- Evaluation schemes aligned with future usage.
- Treating preprocessing, feature generation, and modeling as a single pipeline.
- Monitoring for drift, leakage, and error concentration.

## What Doesn’t
- Optimizing proxy metrics disconnected from decisions.
- Assuming simpler models are inherently safer.
- Treating cross-validation as a guarantee rather than a diagnostic.
- Separating data preparation from modeling responsibility.

## Evaluation Failure That Repeats

![Eval failure diagram](images/useful-things-ml.png)

Typical sequence:
- Dataset collected in full.
- Missing values imputed globally.
- Features normalized using global statistics.
- Train/test split applied afterward.

Outcome:
- Test contamination.
- Inflated evaluation metrics.
- Model selection biased toward overfitting.

Root cause:
Preprocessing treated as preliminary work rather than part of the model.

Correction:
- Split first.
- Fit preprocessing on training data only.
- Freeze transforms for validation and test.
- Evaluate the full pipeline.

## Takeaway
Most ML failures are predictable. They originate from evaluation design and problem framing, not from algorithm choice.
