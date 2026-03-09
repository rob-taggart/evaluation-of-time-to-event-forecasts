# evaluation-of-time-to-event-forecasts
Code and data to generate results and plots from the paper "On the evaluation of time-to-event, survival time and first passage time forecasts" by Robert J. Taggart, Nicholas Loveday and Simon Louis.


- `synthetic_experiment.ipynb` contains code used to generate results for the synthetic experiment (Sections 2 and 3 of the paper).
- `hnv.ipynb` contains code used to generate results for hydrological prediction in the Hawkesbury-Nepean Valley (HNV) (Section 4.1 of the paper).
- `kurnel_timeseries.ipynb` and `kurnell_recalibration.ipynb` contain code used to generate results for the Kurnell time-to-strong-wind forecasts (Section 4.2 of the paper).
- `tte.py` contains code for scoring rules used in Sections 2 and 3 of the paper. Of particular interest is code for the threshold-weighted CRPS when the predictive distribution is a gamma distribution.
- `data` contains forecast and observation data used by the notebooks.

Acknowledgement: This code was reviewed by Nicholas Loveday.
