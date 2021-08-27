# Activity Schedules + Deep Generative Learning (Variational Auto-Encoders)

Variational Auto-Encoder --> VAE

Activity Model --> ACT

Variational Auto-Encoder + Activity Model --> VAEACT

~\~> CAVEAT

Early experiments using VAEs for synthesising realistic activity sequences.

Activity sequence modelling is a core component of modern agent-based simulations that seek to model human decision making in a population. If you come from an economics or statistics background you can think of this as discrete choice modelling on steroids. If you come from the pure sciences you can think of this as applied agent based modelling for humans. If you come from the real world this is sim-city but you're in it too.

Existing methodologies for modelling activity sequence tend to be either (i) nested descrete choice models or (ii) sequence schedulers. In either case, significant complexity is required to vaguelly approach observed population hetrogeneity. In essence this work seeks to adopt modern ML deep learning architectures to enable generation. Even ignoring potential quality improvements from these techniques, this work shows the convenience of more flexible data representations with deep ML.

## The data

The demo currently runs on fake data generated via some bespoke Marcov Chains (data_gen.ipynb). The synthetic data is deliberately generated with three subpopulations - full time workers, in education and part time workers.

## The demo

caveat.ipynb shows how a VAE can be used for synthesis. The notebook shows how the subpopulations are still recoverable and represented post latent layer compression. Sampling, interpolating and smoothing are also demonstrated.

## Notes

The current VAE uses an "image" based data representation and CNN for convenience but it would also be awesome to explore sequence based representations.

These markov chains should be generalised and moved into teh CityChef project.

It would be convenient to ingest a generic data format (eg CityChef or PAM).
