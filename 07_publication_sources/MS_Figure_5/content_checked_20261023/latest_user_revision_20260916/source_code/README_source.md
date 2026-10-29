# Figure 5 A/B/C redraw

This folder contains standalone SVG redraws for the current manuscript Figure 5 panels A-C.

- `Figure5_panel_A_redraw.svg`: fixed width:height ratio 227:135; y-axis labels are shown every 2 entities; x/y tick values and axis labels are enlarged. The trajectory/structure layer uses the highest-resolution archived layer available in the handoff (`1783 x 729` source pixels) and is embedded at approximately `1816 x 1095` pixels instead of the earlier `606 x 365` pixels.
- `Figure5_panel_B_redraw.svg`: natively redrawn from `figure6bc_current_parameter_phase_v1_sensitivity.csv`; bars, confidence intervals, axes, labels, and legend are vector elements.
- `Figure5_panel_C_redraw.svg`: preserves the approved manuscript C-panel surface, scatter points, camera angle, legend, and all data geometry exactly; only the vector axis tick values and axis labels are enlarged.
- `build_figure5_AC_redraw_v2.py`: reproducible redraw script for all three panels.

The current manuscript Figure 5 is provenance-linked to the historical Figure 6 workspace. No stochastic simulation is rerun. Panel B is recalculated from the archived sensitivity table. Panel C uses the approved manuscript vector geometry because the original marching-surface analysis module was not included in the handoff. Panel A uses the archived trajectory/structure image layer because the raw trajectory-rendering source was also not included.
