import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('Bioflow Model')
st.markdown('Bioreactor code that models the concentration of NPEC (non-pathogenic E. coli) that produces GFP (green fluorescent protein) in a bioreactor. It also models the concentration of oxygen and the mass transfer rate. The mass transfer rate increases with agitation and flow rate of air into the bioreactor.')

def derivatives(y, t, p):
    # Unpack the state vector and parameters
    X, S, kla = y; mua, mum, Ks, Xm, b, C, Kp, Ki, Do = p
    # Compute the derivatives
    # First the Logistic - Monod Equation
    dXdt = (mua + (mum * S)/(Ks + S))*(1-X/Xm)*X
    # Then the substrate depletion with bounds on X not to go below 0
    if S < 0:
        dSdt = 0
    else:
        dSdt = -X*b + kla*(C - S)
    # Finally the kla change with bounds not to go below 0 or above 20
    if kla < 0: # or kla > 20:
        dkladt_PI = 0
    else:  
        #dkladt = -Kp*dSdt + Ki*(Do - S) see comment below why only integral control is used
        dkladt_PI = Ki*(Do - S)


    # B. The Feed-Forward part
    # Derivative of the FF law: (b / (C - Do)) * dXdt, This proactively ramps up kla as the cells grow
    if (C - Do) > 0: # Avoid division by zero
        dkladt_FF = (b / (C - Do)) * dXdt
    else:
        dkladt_FF = 0
        
    # Combine them
    if kla > 20:
        dkladt = 0
    else:
        dkladt = dkladt_PI + dkladt_FF

    return [dXdt, dSdt, dkladt]


# --- SIDEBAR & PARAMETERS ---
st.sidebar.title('Parameters')
st.sidebar.markdown('Adjustable parameters.')

# 1. WRAP INPUTS IN A FORM
# This prevents the app from rerunning instantly when you move a slider.
with st.sidebar.form(key='simulation_form'):
    # Sliders
    Do_input = st.slider('Dissolved Oxygen Setpoint (Do)', min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    Ki_input = st.slider('Integral Gain (Ki, 1/hr)', min_value=80.0, max_value=140.0, value=110.0, step=15.0, help='Integral control parameter for disolved oxygen (S)')
    mua_input = st.slider('Anaerobic Rate ($\mu_a$, 1/hr)', min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    mum_input = st.slider('Growth Rate ($\mu_m$, 1/hr)', min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    Ks_input = st.slider('Half Saturation Constant (Ks)', min_value=0.1, max_value=10.0, value = 0.8, step=0.1, help='Substrate concentration when the growth rate is half the maximum')
    b_input = st.slider('Oxygen consumption rate (b, 1/hr)', min_value=0.0, max_value=1000.0, value=100.0, step=5.0)
    
    # The Submit Button
    submit_button = st.form_submit_button(label='Simulate')

    #The Default Button
    defaults_button = st.form_submit_button(label='Reset', help='Click to reset parameters to defaults')

#Reset button used to reset parameter values to defualts
if defaults_button:
    st.session_state.Do_input = 0.8
    Do_input = 0.8
    st.session_state.Ki_input = 110.0
    Ki_input = 110.0


# --- SIMULATION LOGIC ---
# We only run this block if the user hits "Simulate"
if submit_button:
    # Fixed parameter values
    C = 1 
    Xm = 1 
    Kp = 0.2
    
    # Pack parameters with inputs from the form
    p = [mua_input, mum_input, Ks_input, Xm, b_input, C, Kp, Ki_input, Do_input]

    # Initial condition
    X0 = 1e7/5e9 # cells/mL
    S0 = 0.8 # mg/L
    kla0 = 0.2 #1/hr
    y0 = [X0, S0, kla0]

    # Solve ODEs
    t = np.linspace(0, 10, 10000) # 10 hours
    vals = np.zeros((len(t), 3))
    vals[0, :] = y0
    
    for i in range(1, len(t)):
        dXdt, dSdt, dkladt = derivatives(vals[i-1, :], t[i-1], p)
        vals[i, 0] = vals[i-1, 0] + dXdt * (t[i] - t[i-1])
        vals[i, 1] = vals[i-1, 1] + dSdt * (t[i] - t[i-1])
        vals[i, 2] = vals[i-1, 2] + dkladt * (t[i] - t[i-1])

    # Save results to Session State so they persist if the app reruns elsewhere
    st.session_state['data'] = pd.DataFrame({
        'Time': t,
        'S': vals[:, 1],     
        'kla': vals[:, 2],   
        'X/Xm': vals[:, 0]   
    })
    st.session_state['Do_val'] = Do_input
    st.session_state['has_run'] = True


# --- PLOTTING ---
# We check if 'has_run' is true. If not, we skip plotting to avoid the "df not defined" error.
if 'has_run' in st.session_state and st.session_state['has_run']:
    
    # Retrieve data from session state
    df = st.session_state['data']
    Do_val = st.session_state['Do_val']
    
    # Recreate the threshold dataframe for the current Do value
    df_threshold = pd.DataFrame({'val': [Do_val]})


    # Base chart for the main time-series data
    base = alt.Chart(df).encode(
        x=alt.X('Time', title='Time (hours)')
    )
    
    # Left Axis: kla
    left_chart = base.mark_line(color='#1f77b4').encode(
        y=alt.Y('kla', 
                scale=alt.Scale(domain=[0, 20.5]), 
                title='kla (1/hr)')
    )
    
    # Right Axis: S and X/Xm 
    right_lines = base.transform_fold(
        ['S', 'X/Xm'],
        as_=['Variable', 'Value']
    ).mark_line().encode(
        y=alt.Y('Value:Q', 
                scale=alt.Scale(domain=[0, 1]), 
                title='X/Xm (green) and S/C (black)'),
        
        # Colors: S -> Black, X/Xm -> Green
        color=alt.Color('Variable:N', 
                        scale=alt.Scale(domain=['S', 'X/Xm'], 
                                        range=['black', 'green']),
                        legend=alt.Legend(title="Variables", orient='top-right')),
        
        # Dashes: S -> Dash-Dot, X/Xm -> Solid
        strokeDash=alt.StrokeDash('Variable:N', 
                                  scale=alt.Scale(domain=['S', 'X/Xm'], 
                                                  range=[[5, 3, 1, 3], [0]]))
    )
    
    # Right Axis: Do (Threshold): We use df_threshold here instead of 'base'
    do_rule = alt.Chart(df_threshold).mark_rule(
        strokeDash=[5, 5], 
        color='black', 
        opacity=0.5
    ).encode(
        # We must explicitly set the domain to match the right axis [0, 1]
        y=alt.Y('val', scale=alt.Scale(domain=[0, 1]))
    )
    
    # Label for Do
    do_label = alt.Chart(df_threshold).mark_text(
        align='left', baseline='bottom', dx=5, dy=-2
    ).encode(
        y=alt.Y('val', scale=alt.Scale(domain=[0, 1])),
        x=alt.value(0), # Stick to the left side of the chart
        text=alt.value('Do')
    )
    
    # Combine layers: Group the Right Axis components (lines + rule + label)
    right_layer = alt.layer(right_lines, do_rule, do_label)
    
    # Combine Left and Right, resolving the Y scale
    final_chart = alt.layer(
        left_chart,
        right_layer
    ).resolve_scale(
        y='independent'
    ).properties(
        title='Reaction Kinetics Results'
    )
    
    # Render Chart
    st.altair_chart(final_chart, use_container_width=True)


st.markdown('## Understand the Logistic-Monod Growth Model') 

st.markdown('The logistic portion of the growth model models the limits of growth due to the carrying capacity of the bioreactor. The Monod portion models the growth rate as a function of the substrate concentration. The logistic-Monod growth model can be expressed as (Monod portion first and the logistic portion second):')

st.markdown('$\\frac{1}{X}\\frac{dX}{dt} = \\left[ \\mu_a + \\frac{\\mu_{m}S}{K_s + S} \\right] \\left[ 1-\\frac{X}{X_m} \\right]$')


st.markdown('where $\mu_a$ is the NPEC growth rate without substrate (oxygen), $\mu_m$ is the maximum specific growth rate of NPEC with substrate (oxygen), $K_s$ is the Monod constant for substrate (oxygen), $S$ is the substrate (oxygen) concentration, $X$ is the NPEC concentration, and $X_m$ is the carrying capacity of the bioreactor.')
