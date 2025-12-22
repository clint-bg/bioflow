import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title('bioflow model')
st.markdown('Bioreactor code that models the concentration of NPEC (non-pathogenic E. coli) that produces GFP (green fluorescent protein) in a bioreactor. It also models the concentration of oxygen and the mass transfer rate. The mass transfer rate increases with agitation and flow rate of air into the bioreactor.')

st.markdown('## hello')

# Sidebar
st.sidebar.title('parameters')
st.sidebar.markdown('Adjustable parameters.')

