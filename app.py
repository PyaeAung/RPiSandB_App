import streamlit as st
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
from obspy.signal import filter
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gs
import numpy as np
from obspy.taup import TauPyModel
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import BytesIO

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Raspberry Shake Seismograph Plotter")

# --- Helper Functions (Your original functions) ---
cmap = plt.get_cmap('Paired', lut=12)
COLORS = ['#%02x%02x%02x' % tuple(int(col * 255) for col in cmap(i)[:3]) for i in range(12)]
COLORS = COLORS[1:][::2][:-1] + COLORS[::2][:-1]

def plot_arrivals(ax, d1, pt, pb, bottoff, arrs, no_arrs, rayt, lovet, delay, duration):
    y1 = -1; axb, axt = ax.get_ylim()
    for q in range(no_arrs):
        x1 = arrs[q].time
        if delay <= x1 < delay + duration:
            ax.axvline(x=x1 - d1, linewidth=1, linestyle='--', color=COLORS[q % len(COLORS)])
            if y1 < 0 or y1 < axt / 2: y1 = axt * pt
            else: y1 = axt * 0.1 if bottoff else axb * pb
            ax.text(x1 - d1, y1, arrs[q].name, alpha=1, color=COLORS[q % len(COLORS)])
    for name, arrival_time in [('Ray', rayt), ('Love', lovet)]:
        if delay <= arrival_time < delay + duration:
            ax.axvline(x=arrival_time - d1, linewidth=0.5, linestyle='--', color='k')
            if y1 < 0 or y1 < axt / 2: y1 = axt * pt
            else: y1 = axt * 0.1 if bottoff else axb * pb
            ax.text(arrival_time - d1, y1, name, alpha=0.5)

def time2UTC(a, eventTime): return eventTime + a
def uTC2time(a, eventTime): return a - eventTime
def one_over(a): a = np.array(a, dtype=float); near_zero = np.isclose(a, 0); a[near_zero] = np.inf; a[~near_zero] = 1 / a[~near_zero]; return a
inverse = one_over

def plot_noiselims(ax, uplim, downlim):
    axl, axr = ax.get_xlim()
    for mult in [1, 2, 3]: ax.axhline(y=uplim * mult, lw=0.33, color='r', linestyle='dotted'); ax.axhline(y=downlim * mult, lw=0.33, color='r', linestyle='dotted')
    ax.text(axl, uplim * 3, '3SD background', size='xx-small', color='r', alpha=0.5, ha='left', va='bottom')
    ax.text(axl, downlim * 3, '-3SD background', size='xx-small', color='r', alpha=0.5, ha='left', va='top')

def plot_se_noiselims(ax, uplim):
    axl, axr = ax.get_xlim()
    for mult_sq in [1, 4, 9]: ax.axhline(y=uplim * mult_sq, lw=0.33, color='r', linestyle='dotted')
    ax.axhline(y=0, lw=0.33, color='r', linestyle='dotted')
    ax.text(axl, uplim * 9, '3SD background', size='xx-small', color='r', alpha=0.5, ha='left', va='bottom')

def divTrace(tr, n): return tr.__truediv__(n) if n > 0 else [tr]
def fmtax(ax, lim, noneg): ax.xaxis.set_minor_locator(AutoMinorLocator(10)); ax.yaxis.set_minor_locator(AutoMinorLocator(5)); ax.margins(x=0); grid(ax); ax.set_ylim(*(0, lim) if noneg else (-lim, lim)) if lim != 0 else None
def grid(ax): ax.grid(color='dimgray', ls='-.', lw=0.33); ax.grid(color='dimgray', which='minor', ls=':', lw=0.33)
def sax(secax, tix, tlabels): secax.set_xticks(ticks=tix); secax.set_xticklabels(tlabels, size='small', va='center_baseline'); secax.xaxis.set_minor_locator(AutoMinorLocator(10))

# --- Streamlit Sidebar ---
st.sidebar.title("Seismograph Plot Generator")
st.sidebar.markdown("ငလျင်နှင့် R-Shake Station အချက်အလက်များကို ထည့်သွင်းပါ။")
event_time_str = st.sidebar.text_input("Event Time (UTC)", "2025-07-03T00:40:40", help="Format: YYYY-MM-DDTHH:MM:SS")
latE = st.sidebar.number_input("Latitude (°N)", -90.0, 90.0, 21.6, format="%.4f")
lonE = st.sidebar.number_input("Longitude (°E)", -180.0, 180.0, 95.5, format="%.4f")
depth = st.sidebar.number_input("Depth (km)", 0.0, 1000.0, 17.0, 1.0)
mag = st.sidebar.number_input("Magnitude", 0.0, 10.0, 4.4, 0.1)
eventID = st.sidebar.text_input("Event ID", "rs2025mxzgen")
locE = st.sidebar.text_input("Event Location Description", "Myingyan: 20 km, 233°")
nw = st.sidebar.text_input("Network Code", "AM")
stn = st.sidebar.text_input("Station Code", "R7107")
ch = st.sidebar.selectbox("Channel Code", ['*HZ', '*HNZ', 'EHZ', 'SHZ'], 0)
delay = st.sidebar.number_input("Plot Start Delay from Event (seconds)", 0, 3600, 0)
duration = st.sidebar.number_input("Plot Duration (seconds)", 10, 7200, 200)
f_min = st.sidebar.number_input("Min Frequency (Hz)", 0.01, 49.0, 0.7, 0.1)
f_max = st.sidebar.number_input("Max Frequency (Hz)", 0.02, 50.0, 8.0, 0.1)
filt = [f_min * 0.99, f_min, f_max, f_max * 1.01]
calcmL = st.sidebar.checkbox("Calculate ML Estimates", value=True)
plot_envelopes = st.sidebar.checkbox("Plot Envelopes on Traces", False)
allphases = st.sidebar.checkbox("Plot All Phases (not just in time window)", True)
bnst = st.sidebar.number_input("Noise Sample Start (sec before event)", 0, 10000, 0)
bne = st.sidebar.number_input("Noise Sample End (sec after event)", 0, 10000, 600)
bnsamp = st.sidebar.number_input("BN Sample size (s)", 1, 100, 15)
notes1 = st.sidebar.text_area("Notes", "")
twitter_handle = st.sidebar.text_input("Your Twitter Handle", "@AlanSheehan18")

# --- Main App ---
st.title("Raspberry Shake - Earthquake Data Visualization")

if st.sidebar.button("Generate Plot"):
    try: eventTime = UTCDateTime(event_time_str)
    except Exception: st.error("Invalid Event Time format."); st.stop()

    with st.spinner('Working...'):
        try:
            # STEP 1: DATA & CORE CALCS
            rs = Client('https://data.raspberryshake.org/')
            inv = rs.get_stations(network=nw, station=stn, level='RESP')
            sta = next(s for s in inv[0] if s.is_active(time=eventTime))
            latS, lonS, eleS = sta.latitude, sta.longitude, sta.elevation
            sab = 'Raspberry Shake and Boom' if any(c.code == 'HDF' for c in sta.channels) else 'Raspberry Shake'
            
            latSrad, lonSrad = math.radians(latS), math.radians(lonS)
            latErad, lonErad = math.radians(latE), math.radians(lonE)
            lon_diff = abs(lonSrad - lonErad)
            great_angle_rad = math.acos(math.sin(latErad) * math.sin(latSrad) + math.cos(latErad) * math.cos(latSrad) * math.cos(lon_diff))
            great_angle_deg = math.degrees(great_angle_rad)
            distance = great_angle_rad * 6371 * math.pi / 180

            # STEP 2: MAP PROJECTION
            if great_angle_deg < 5: sat_height = 1_000_000
            elif great_angle_deg < 25: sat_height = 10_000_000
            else: sat_height = 100_000_000
            latC, lonC = (latE + latS) / 2, (lonE + lonS) / 2
            if abs(lonE - lonS) > 180: lonC += 180
            projection = ccrs.NearsidePerspective(central_latitude=latC, central_longitude=lonC, satellite_height=sat_height)
            mtext = f'Satellite Viewing Height = {int(sat_height/1000)} km.'
            if great_angle_deg > 150: projection = ccrs.Orthographic(central_latitude=latC, central_longitude=lonC); mtext = 'Orthographic projection.'
            projection._threshold /= 20

            # STEP 3: WAVEFORM & ANALYSIS
            start, end = eventTime + delay, eventTime + delay + duration
            trace0 = rs.get_waveforms(nw, stn, '00', ch, start, end); trace0.merge(method=0, fill_value='latest').detrend(type='demean')
            rawtrace = trace0.copy()
            outdisp = rawtrace.copy().remove_response(inventory=inv, pre_filt=filt, output='DISP', water_level=60)
            outvel = rawtrace.copy().remove_response(inventory=inv, pre_filt=filt, output='VEL', water_level=60)
            outacc = rawtrace.copy().remove_response(inventory=inv, pre_filt=filt, output='ACC', water_level=60)
            jerk = outacc.copy().differentiate()
            outSE = outvel.copy(); outSE[0].data = outSE[0].data**2 / 2
            
            bnstart, bnend = eventTime - bnst, eventTime + bne
            bn0 = rs.get_waveforms(nw, stn, '00', ch, bnstart, bnend); bn0.merge(method=0, fill_value='latest').detrend(type='demean')
            bndisp = bn0.copy().remove_response(inventory=inv, pre_filt=filt, output='DISP', water_level=60)
            bnvel = bn0.copy().remove_response(inventory=inv, pre_filt=filt, output='VEL', water_level=60)
            bnacc = bn0.copy().remove_response(inventory=inv, pre_filt=filt, output='ACC', water_level=60)
            bnjerk = bnacc.copy().differentiate()
            
            bns = int((bne + bnst) / bnsamp)
            bnd, bnv, bna, bnj = divTrace(bndisp[0], bns), divTrace(bnvel[0], bns), divTrace(bnacc[0], bns), divTrace(bnjerk[0], bns)
            bndispstd = np.min([abs(t.std()) for t in bnd])
            bnvelstd = np.min([abs(t.std()) for t in bnv])
            bnaccstd = np.min([abs(t.std()) for t in bna])
            bnjstd = np.min([abs(t.std()) for t in bnj])
            bnsestd = bnvelstd**2 / 2

            model = TauPyModel(model='iasp91')
            arrs = model.get_travel_times(depth, great_angle_deg)
            no_arrs = len(arrs)
            rayt, lovet = distance / 3.2206, distance / 4.2941
            qenergy = 10**(1.5 * mag + 4.8)
            disp_max, vel_max, acc_max, jmax = outdisp[0].max(), outvel[0].max(), outacc[0].max(), jerk[0].max()
            se_max = vel_max**2 / 2
            
            if calcmL:
                mLDv = np.log10(abs(disp_max/1e-6))+2.234*np.log10(distance)-1.199
                mLVv = np.log10(abs(vel_max/1e-6))+2.6235*np.log10(distance)-3.415
                mLAv = np.log10(abs(acc_max/1e-6))+3.146*np.log10(distance)-6.154

            # STEP 4: PLOTTING
            fig = plt.figure(figsize=(20, 14), dpi=150)
            # ... (Layout setup remains the same) ...
            gsouter = gs.GridSpec(1, 3, width_ratios=[10.5, 2.5, 7], wspace=0.1)
            gsleft = gs.GridSpecFromSubplotSpec(3, 1, subplot_spec=gsouter[0], height_ratios=[4, 1, 1], hspace=0.15)
            gsright = gs.GridSpecFromSubplotSpec(4, 1, subplot_spec=gsouter[2], height_ratios=[2, 2, 1, 1], hspace=0.1)
            gsl2 = gs.GridSpecFromSubplotSpec(4, 1, subplot_spec=gsleft[0], hspace=0)
            ax1, ax2, ax3, ax9 = [fig.add_subplot(gsl2[i]) for i in range(4)]
            ax6, ax4 = fig.add_subplot(gsleft[1]), fig.add_subplot(gsleft[2])
            ax7 = fig.add_subplot(gsright[0], polar=True)
            ax8 = fig.add_subplot(gsright[1], projection=projection)
            ax5, ax10 = fig.add_subplot(gsright[2]), fig.add_subplot(gsright[3])
            
            # --- Left Side Plots ---
            times = trace0[0].times(reftime=eventTime)
            ax1.plot(times, outdisp[0].data, lw=1, color='b'); ax1.set_ylabel("Displacement, m", size='small')
            ax2.plot(times, outvel[0].data, lw=1, color='g'); ax2.set_ylabel("Velocity, m/s", size='small')
            ax3.plot(times, outacc[0].data, lw=1, color='r'); ax3.set_ylabel("Acceleration, m/s²", size='small')
            ax9.plot(times, jerk[0].data, lw=1, color='purple'); ax9.set_ylabel("Jerk, m/s³", size='small')
            
            for ax in [ax1, ax2, ax3, ax9]: fmtax(ax, 0, False); plot_noiselims(ax, bndispstd if ax==ax1 else bnvelstd if ax==ax2 else bnaccstd if ax==ax3 else bnjstd, -(bndispstd if ax==ax1 else bnvelstd if ax==ax2 else bnaccstd if ax==ax3 else bnjstd))
            ax9.set_xlabel('Seconds after Event, s', size='small', alpha=0.5)
            for ax in [ax1, ax2, ax3]: ax.tick_params(axis='x', colors='w')
            
            ax6.plot(times, outSE[0].data, lw=1, color='g', linestyle=':'); fmtax(ax6, 0, True); plot_se_noiselims(ax6, bnsestd)
            ax6.set_ylabel('Specific Energy, J/kg', size='small'); ax6.set_xlabel('Seconds after Event, s', size='small', alpha=0.5)
            
            ax4.specgram(x=rawtrace[0], NFFT=256, noverlap=128, Fs=100, cmap='viridis'); ax4.set_yscale('log'); ax4.set_ylim(0.5, 50); ax4.grid(True)
            ax4.axhline(y=filt[1], lw=1, color='r', linestyle='dotted'); ax4.axhline(y=filt[2], lw=1, color='r', linestyle='dotted')
            ax4.set_ylabel("Frequency, Hz", size='small'); ax4.set_xlabel('Seconds after Start of Trace, s', size='small', alpha=0.5)
            
            for ax in [ax1, ax2, ax3, ax9]: plot_arrivals(ax, 0, 0.8, 0.95, False, arrs, no_arrs, rayt, lovet, delay, duration)
            plot_arrivals(ax4, delay, 0.6, 1.8, False, arrs, no_arrs, rayt, lovet, delay, duration)
            plot_arrivals(ax6, 0, 0.9, 1, True, arrs, no_arrs, rayt, lovet, delay, duration)
            
            # --- Right Side Plots ---
            ax8.coastlines(resolution='110m'); ax8.stock_img(); ax8.gridlines()
            ax8.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', edgecolor='gray', facecolor='none'))
            ax8.plot(lonS, latS, color='red', marker='v', markersize=12, transform=ccrs.Geodetic())
            ax8.plot(lonE, latE, color='yellow', marker='*', markersize=20, transform=ccrs.Geodetic())
            ax8.plot([lonS, lonE], [latS, latE], color='blue', linestyle='--', transform=ccrs.Geodetic())
            
            pphases = [a.name for a in arrs if allphases or (delay <= a.time < delay + duration)]
            arrivals = model.get_ray_paths(depth, great_angle_deg, phase_list=pphases)
            arrivals.plot_rays(plot_type='spherical', ax=ax7, fig=fig, phase_list=pphases, show=False, legend=True)
            ax7.text(great_angle_rad, 7000, stn, ha='center', va='center', alpha=.7, size='small', rotation=-great_angle_deg)
            ax7.text(0, 0, 'Solid\ninner\ncore', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
            
            nfft = 8192 if duration >= 82 else int(duration * 100)
            ax5.psd(x=outdisp[0], NFFT=nfft, Fs=100, color='b', label='Disp')
            ax5.psd(x=outvel[0], NFFT=nfft, Fs=100, color='g', label='Vel')
            ax5.psd(x=outacc[0], NFFT=nfft, Fs=100, color='r', label='Acc')
            ax5.psd(x=jerk[0], NFFT=nfft, Fs=100, color='purple', label='Jerk')
            ax5.set_xscale('log'); ax5.legend(frameon=False, fontsize='x-small'); grid(ax5)
            ax5.axvline(x=filt[1], c='r', ls=':'); ax5.axvline(x=filt[2], c='r', ls=':')
            ax5.set_ylabel("PSD, dB", size='small'); ax5.set_xlabel('Frequency, Hz', size='small', alpha=0.5, labelpad=-9)
            secax_x5 = ax5.secondary_xaxis('top', functions=(one_over, inverse)); secax_x5.set_xlabel('Period, s', size='small', alpha=0.5, labelpad=-9)
            
            xfft = np.fft.rfftfreq(outvel[0].data.size, d=1/100)
            ax10.plot(xfft, abs(np.fft.rfft(outvel[0].data)), color='g', lw=1, label='Vel')
            ax10.legend(frameon=False, fontsize='x-small'); fmtax(ax10, 0, True)
            ax10.axvline(x=filt[1], c='r', ls=':'); ax10.axvline(x=filt[2], c='r', ls=':')
            ax10.set_xlim(filt[1] * .75, filt[2] * 1.25 if filt[2] >= 4 else filt[2] + 1)
            ax10.set_ylabel("FFT Spectrum", size='small'); ax10.set_xlabel('Frequency, Hz', size='small', alpha=0.5)
            
            # --- All Text Annotations ---
            fig.suptitle(f"M{mag} Earthquake - {locE} - {eventTime.strftime('%d/%m/%Y %H:%M:%S UTC')}", weight='black', color='b', size='x-large')
            fig.text(0.05, 0.95, f"Filter: {filt[1]:.2f} to {filt[2]:.2f}Hz")
            fig.text(0.7, 0.95, f'Event ID: {eventID}')
            fig.text(0.95, 0.95, f'Station: {nw}.{stn}.00.{ch}', ha='right', size='large')
            fig.text(0.95, 0.935, sab, color='r', ha='right')
            fig.text(0.95, 0.92, 'Oberon Citizen Science Network', size='small', ha='right')
            
            y_pos = 0.905
            for tag in ['#ShakeNet', '@raspishake', twitter_handle, '@matplotlib', '#Python', '#CitizenScience', '#Obspy', '#Cartopy']:
                fig.text(0.98, y_pos, tag, ha='right'); y_pos -= 0.015

            try:
                rsl = plt.imread("RS logo.png")
                newaxr = fig.add_axes([0.935, 0.915, 0.05, 0.05], anchor='NE', zorder=-1)
                newaxr.imshow(rsl); newaxr.axis('off')
            except FileNotFoundError: st.warning("RS logo.png not found.")

            fig.text(0.91, 0.37, f'NOTES: {notes1}', rotation=90)
            fig.text(0.51, 0.85, f'Max D = {disp_max:.3E} m', size='small', rotation=90, va='center', color='b')
            fig.text(0.51, 0.72, f'Max V = {vel_max:.3E} m/s', size='small', rotation=90, va='center', color='g')
            fig.text(0.51, 0.58, f'Max A = {acc_max:.3E} m/s²', size='small', rotation=90, va='center', color='r')
            fig.text(0.51, 0.455, f'Max Jerk = {jmax:.3E} m/s³', size='small', rotation=90, va='center', color='purple')
            fig.text(0.51, 0.28, f'Max SE = {se_max:.3E} J/kg', size='small', rotation=90, va='center', color='g')

            if calcmL:
                fig.text(0.52, 0.85, f'MLDv = {mLDv:.1f} +/- 1.40', size='small', rotation=90, va='center')
                fig.text(0.52, 0.72, f'MLVv = {mLVv:.1f} +/- 1.56', size='small', rotation=90, va='center')
                fig.text(0.52, 0.58, f'MLAv = {mLAv:.1f} +/- 1.89', size='small', rotation=90, va='center')

            fig.text(0.53, 0.33, 'Background Noise:', size='small')
            fig.text(0.53, 0.32, f'Displacement SD = {bndispstd:.3E} m', color='b', size='small')
            fig.text(0.53, 0.29, f'Velocity SD = {bnvelstd:.3E} m/s', color='g', size='small')
            fig.text(0.53, 0.26, f'Acceleration SD = {bnaccstd:.3E} m/s²', color='r', size='small')
            fig.text(0.53, 0.23, f'Jerk SD = {bnjstd:.3E} m/s³', color='purple', size='small')
            fig.text(0.53, 0.20, f'Specific Energy SD = {bnsestd:.3E} J/kg', color='g', size='small')
            fig.text(0.53, 0.17, 'BN parameters:', size='small')
            fig.text(0.53, 0.16, f'Start: Event time - {bnst} s.', size='small')
            fig.text(0.53, 0.15, f'End: Event time + {bne} s.', size='small')
            fig.text(0.53, 0.14, f'BN Sample size = {bnsamp} s.', size='small')
            
            y2 = 0.93; dy = 0.008
            fig.text(0.53, y2, 'Phase', size='x-small'); fig.text(0.565, y2, 'Time', size='x-small'); fig.text(0.6, y2, 'UTC', size='x-small')
            for i in range(no_arrs):
                y2 -= dy; alf = 1.0 if delay <= arrs[i].time < (delay+duration) else 0.53
                fig.text(0.53, y2, arrs[i].name, size='x-small', alpha=alf)
                fig.text(0.565, y2, f'{arrs[i].time:.3f}s', size='x-small', alpha=alf)
                fig.text(0.6, y2, (eventTime + arrs[i].time).strftime('%H:%M:%S'), size='x-small', alpha=alf)
            
            y2 = 0.7; dy = 0.008
            fig.text(0.98, y2, 'Phase Key', size='small', ha='right')
            pkey = ['P: compression wave', 's: strictly upward shear wave', 'c: reflection off outer core', 'i: reflection off inner core']
            for item in pkey: y2 -= dy; fig.text(0.98, y2, item, size='x-small', ha='right')

            plt.subplots_adjust(hspace=0.3, wspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.92)
            st.pyplot(fig)
            st.success("Plot generated successfully!")
            buf = BytesIO(); fig.savefig(buf, format="png")
            st.download_button("Download Plot as PNG", buf.getvalue(), f"M{mag}_{eventID}_{stn}.png", "image/png")
            
        except Exception as e:
            st.error(f"An error occurred: {e}"); st.exception(e)