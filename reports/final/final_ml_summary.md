# NariSafe Final ML Summary

## Final selected model

**No-location HistGradientBoosting with high-risk threshold tuning**

## Why this model was selected

This model was selected because it performed best on the realistic group-city evaluation setup after removing direct location identity features like city, latitude, and longitude.

The final model uses contextual, crime, and infrastructure features instead of direct city identity.

## Final model path

`models/no_location_baselines/best_no_location_threshold_model.joblib`

## Selected high-risk threshold

`0.2`

## Removed features

- `city`
- `latitude`
- `longitude`

## Reason for removing location identity

City, latitude, and longitude were removed to reduce direct location memorization and test whether the model can learn general safety-context patterns from time, area type, complaint severity, crime statistics, police access, transport access, lighting, and crowd context.

## Primary evaluation split

`group_city`

Group-city split is treated as the main benchmark because it tests the model on cities not seen during training.

## Final threshold-tuned metrics

- `threshold`: `0.2`
- `accuracy`: `0.833819`
- `macro_f1`: `0.766774`
- `weighted_f1`: `0.831308`
- `high_precision`: `0.936561`
- `high_recall`: `0.494709`
- `high_f1`: `0.647432`

## Important limitations

- `risk_level` is a rule-based proxy label, not real incident-level ground truth.
- This is a risk-awareness prototype, not a guaranteed crime prediction system.
- NCRB/OpenCity crime data is city-level, not hyperlocal incident-level data.
- OSM infrastructure features may be incomplete, especially street-light data.
- Prediction probabilities should be treated as model confidence, not real-world probability of crime.

## Final recommendation

Use the threshold-tuned no-location model for the final demo and API. Display predictions as contextual risk-awareness levels, not as exact crime probabilities.