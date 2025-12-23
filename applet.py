import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('Bioflow Model')
st.markdown('Bioreactor code that models the concentration of NPEC (non-pathogenic E. coli) that produces GFP (green fluorescent protein) in a bioreactor. It also models the concentration of oxygen and the mass transfer rate. The mass transfer rate increases with agitation and flow rate of air into the bioreactor. Click the simulate button in the left sidebar.')

st.info('Use this simple simulation to better understand how parameters like (1) the limitations of the mass transfer of oxygen (kla value), (3) the starting concentration of bacteria (X/Xm), (3) half saturation constant, (4) and other parameters impact the growth OF NPEC.')

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
    Ki_input = st.slider('Integral Gain (Ki, 1/hr)', min_value=80.0, max_value=140.0, value=110.0, step=2.5, help='Integral control parameter for disolved oxygen (S) with default value of 110')
    mua_input = st.slider('Anaerobic Rate ($\mu_a$, 1/hr)', min_value=0.0, max_value=1.0, value=0.1, step=0.05, help='Default value of 0.1')
    mum_input = st.slider('Growth Rate ($\mu_m$, 1/hr)', min_value=0.0, max_value=10.0, value=1.4, step=0.1, help='Default value of 1.4')
    Ks_input = st.slider('Half Saturation Constant (Ks)', min_value=0.1, max_value=10.0, value = 0.8, step=0.1, help='Substrate concentration when the growth rate is half the maximum with default value of 0.8')
    b_input = st.slider('Oxygen Consumption Rate (b, 1/hr)', min_value=0.0, max_value=1000.0, value=100.0, step=5.0, help='Default value of 100')
    S0_input = st.slider('Initial Oxygen Fraction (S0)', min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    X0_input = st.number_input('Initial X/Xm', min_value=1e-6, max_value=5e-2, value=2e-3,help='Defalut value of 2e-3',format="%.3e") 
    
    # The Submit Button
    submit_button = st.form_submit_button(label='Simulate')

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
    kla0 = 0.2 #1/hr
    y0 = [X0_input, S0_input, kla0]

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

st.markdown('The Logistic-Monod model is commonly used to describe microbial growth. It is a hybrid model with the Monod portion describing the growth rate as a function of the substrate concentration. The logistic portion limits the growth due to the carrying capacity of the bioreactor. The logistic-Monod growth model can be expressed as (Monod portion in the first set of brackets and the logistic portion second):')

st.markdown('$\\frac{1}{X}\\frac{dX}{dt} = \\left[ \\mu_a + \\frac{\\mu_{m}S}{K_s + S} \\right] \\left[ 1-\\frac{X}{X_m} \\right]$')


st.markdown('where $\mu_a$ is the NPEC growth rate without substrate (oxygen), $\mu_m$ is the maximum specific growth rate of NPEC with substrate (oxygen), $K_s$ is the Monod constant for substrate (oxygen), $S$ is the substrate (oxygen) concentration, $X$ is the NPEC concentration, and $X_m$ is the carrying capacity of the bioreactor. Other Monod like factors can be added to simulate other required substrates like glucose.')

st.markdown('## Understand transfer and consumption of the substrate (oxygen)')
st.markdown('The substrate (oxygen) concentration changes as a function of time based on the number of NPEC cells and the mass transfer rate. The mass transfer rate is a simplification of what really happens as the rate increases with agitation and flow rate of air into the bioreactor. The change in substrate (oxygen) concentration can be expressed as:')

st.markdown('$\\frac{dS}{dt} = -X\cdot b + k_{La}(C - S)$')

st.markdown('where $X$ is the NPEC concentration, $b$ is the consumption rate of substrate by the NPEC, $C$ is the max oxygen concentration possible, and $k_{La}$ is the mass transfer coefficient.')

st.markdown('## Understand the $k_{La}$ parameter')
st.markdown('The $k_{La}$ is the mass transfer coefficient that increases with agitation and flow rate of air into the bioreactor. However, instead of modeling agitation and flow rate directly, we will just use a proportional and integral (PI) controller to model the changing $k_{La}$. An increasing $k_{La}$ means that more oxygen can be transferred (dissolved) into the fluid. That PI control is:')

st.markdown('$\\frac{dk_{La}}{dt} = -K_p \\frac{dS}{dt} + K_i(D_o - S)$')
st.markdown('where $K_p$ is the proportional gain and $K_i$ is the integral gain. The $D_o$ is the desired oxygen concentration (set point). This equation is just the derivative of the PI control equation.')

st.markdown('With proportional control and the value of S above the setpoint, the controller sees the rate of oxygen drop and tries to anticipate the loss by increasing $k_{La}$ immediately, even though you are currently above the setpoint. This results in a positive change in $k_{La}$ which is unexpected. For bioreactors with noisy consumption rates or when you want the controller to respond strictly to the level of oxygen (not the rate of change), an Integral-only controller (or a very heavily detuned PI) is often preferred.')

st.markdown('Without feed forward control with purely integral control, it can take a while for the integral term to wind up. Feed forward helps with that to calculate how much the oxygen will be needed and adjust kla immediately.')

st.markdown('### Feed forward control')
st.markdown('Rate of consumption by the cells equaling the rate of transfer is:')
st.markdown('$k_{La} = \\frac{bX}{(C-S)}$')

st.markdown('And assuming that the controller does its job then S can be replaced with $D_o$ to get:')
st.markdown('$k_{La} = \\frac{bX}{(C-D_o)}$')
st.markdown('and the derivative of this equation is:')
st.markdown('$\\frac{dk_{La}}{dt} = \\frac{b}{C-D_o}$')

st.markdown('## Solve the Coupled Differential Equations')
st.markdown('Those 3 coupled differential equations are solved simultaneously with a numerical algorithm (in this case using Eulers method). The three coupled equations are:')
st.markdown('$\\frac{1}{X}\\frac{dX}{dt} = \\left[ \\mu_a + \\frac{\\mu_{m}S}{K_s + S} \\right] \\left[ 1-\\frac{X}{X_m} \\right]$')
st.markdown('$\\frac{dS}{dt} = -X\cdot b + k_{La}(C - S)$')
st.markdown('$\\frac{dk_{La}}{dt} = \\frac{b}{C-D_o} + K_i(D_o - S)$')
st.markdown(f"You can download the Jupyter ipynb workbook file here [{"JupyterNotebookFile"}]({"https://github.com/clint-bg/bioflow/blob/main/bioreactormodel.ipynb"})")



