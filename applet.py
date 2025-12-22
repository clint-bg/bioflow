import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('bioflow model')
st.markdown('Bioreactor code that models the concentration of NPEC (non-pathogenic E. coli) that produces GFP (green fluorescent protein) in a bioreactor. It also models the concentration of oxygen and the mass transfer rate. The mass transfer rate increases with agitation and flow rate of air into the bioreactor.')

st.markdown('## Understand the Logistic-Monod Growth Model') 

st.markdown('The logistic portion of the growth model models the limits of growth due to the carrying capacity of the bioreactor. The Monod portion models the growth rate as a function of the substrate concentration. The logistic-Monod growth model can be expressed as (Monod portion first and the logistic portion second):')

st.markdown('$$ \frac{1}{X}\frac{dX}{dt} = \left[ \mu_a + \frac{\mu_{m}S}{K_s + S} \right] \left[ 1-\frac{X}{X_m} \right] $$')

st.markdown('where $\mu_a$ is the NPEC growth rate without substrate (oxygen), $\mu_m$ is the maximum specific growth rate of NPEC with substrate (oxygen), $K_s$ is the Monod constant for substrate (oxygen), $S$ is the substrate (oxygen) concentration, $X$ is the NPEC concentration, and $X_m$ is the carrying capacity of the bioreactor.')



# Sidebar
st.sidebar.title('parameters')
st.sidebar.markdown('Adjustable parameters.')

