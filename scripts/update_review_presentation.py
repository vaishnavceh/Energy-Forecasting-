import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import os

def update_presentation():
    input_path = r"C:\Users\vaish\OneDrive\Desktop\EEE_First_Review_EEE_ER_Presentation_Template (1)_BACKUP.pptx"
    output_path = r"C:\Users\vaish\OneDrive\Desktop\EEE_First_Review_EEE_ER_Presentation_Template (1).pptx"
    fig1_path = r"C:\Users\vaish\.gemini\antigravity\brain\c839cace-1fde-4b66-b314-8d80f369b34f\scratch\slide3_evidence.png"
    fig2_path = r"C:\Users\vaish\.gemini\antigravity\brain\c839cace-1fde-4b66-b314-8d80f369b34f\scratch\slide15_result.png"

    prs = pptx.Presentation(input_path)
    print(f"Loaded presentation with {len(prs.slides)} slides.")

    def set_text(sh, text, font_size=Pt(11), bold=False, font_name="Times New Roman", align=None, color=None):
        if not sh:
            return
        tf = getattr(sh, 'text_frame', None)
        if not tf:
            return
        tf.word_wrap = True
        tf.text = text
        for p in tf.paragraphs:
            p.font.name = font_name
            if font_size:
                p.font.size = font_size
            p.font.bold = bold
            if color:
                p.font.color.rgb = color
            if align is not None:
                p.alignment = align

    def set_paragraphs(sh, lines, font_size=Pt(10), bold=False, font_name="Times New Roman", align=None, color=None):
        if not sh:
            return
        tf = getattr(sh, 'text_frame', None)
        if not tf:
            return
        tf.word_wrap = True
        tf.text = lines[0] if lines else ""
        p0 = tf.paragraphs[0]
        p0.font.name = font_name
        if font_size:
            p0.font.size = font_size
        p0.font.bold = bold
        if color:
            p0.font.color.rgb = color
        if align is not None:
            p0.alignment = align

        for line in lines[1:]:
            p = tf.add_paragraph()
            p.text = line
            p.font.name = font_name
            if font_size:
                p.font.size = font_size
            p.font.bold = bold
            if color:
                p.font.color.rgb = color
            if align is not None:
                p.alignment = align

    def get_shape(slide, name):
        for s in slide.shapes:
            if s.name == name:
                return s
        return None

    def get_group_shape(group, name):
        for s in group.shapes:
            if s.name == name:
                return s
            if s.shape_type == pptx.enum.shapes.MSO_SHAPE_TYPE.GROUP:
                res = get_group_shape(s, name)
                if res:
                    return res
        return None

    # =========================================================================
    # SLIDE 1: Title Slide (1 / 20)
    # =========================================================================
    print("Updating Slide 1...")
    s1 = prs.slides[0]
    sh_dom = get_shape(s1, "TextBox 12")
    if sh_dom:
        set_text(sh_dom, "Broad Research / Application Domain: Power Systems, Smart Grids & Deep Learning (Short-Term Load Forecasting)", font_size=Pt(11.5), bold=True)
    
    sh_guide = get_shape(s1, "TextBox 54")
    if sh_guide:
        guide_lines = [
            "Group No\t\t: ER27A15",
            "Project Guide\t: Dr.Anantha padmanabhan N K, Department of Computer Science Engineering, TKMCE, Kollam",
            "Co-Guide (s)\t\t: Prof.Amina Shahjahan, Assistant Professor, Department of EEE, TKMCE, Kollam"
        ]
        set_paragraphs(sh_guide, guide_lines, font_size=Pt(9.5), bold=False)

    sh_prog = get_shape(s1, "TextBox 55")
    if sh_prog:
        set_text(sh_prog, "Programme: □ EEE   ■ ER                                                                                                Academic Year: 2026–27", font_size=Pt(10.5), bold=True)

    g58 = get_shape(s1, "Group 58")
    if g58:
        # Fix typo in Vaishnav's name
        sh_vaish = get_group_shape(g58, "Rectangle 50")
        if sh_vaish:
            set_text(sh_vaish, "VAISHNAV VENUGOPAL", font_size=Pt(10.5), bold=True)
        # Primary responsibilities
        r1 = get_group_shape(g58, "TextBox 29")
        if r1: set_text(r1, "Literature Survey, Model Architecture & Deep Learning (DeepDeFF)", font_size=Pt(9))
        r2 = get_group_shape(g58, "TextBox 37")
        if r2: set_text(r2, "Smart Meter Data Ingestion, Data Cleaning & Preprocessing Pipeline", font_size=Pt(9))
        r3 = get_group_shape(g58, "TextBox 45")
        if r3: set_text(r3, "Feature Engineering, Cyclical Transforms & Statistical Features", font_size=Pt(9))
        r4 = get_group_shape(g58, "TextBox 53")
        if r4: set_text(r4, "Large-Scale Parquet Compression, Model Training & Benchmarking", font_size=Pt(9))

    # =========================================================================
    # SLIDE 2: Project At A Glance (2 / 20)
    # =========================================================================
    print("Updating Slide 2...")
    s2 = prs.slides[1]
    g70 = get_shape(s2, "Group 70")
    if g70:
        t19 = get_group_shape(g70, "TextBox 19")
        if t19: set_text(t19, "Short-term electricity load in smart grids exhibits severe volatility, multi-scale temporal dependencies, and non-linear consumption patterns across thousands of consumers, leading to high forecasting error and grid operational inefficiency.", font_size=Pt(9.5))
        
        t23 = get_group_shape(g70, "TextBox 23")
        if t23: set_text(t23, "A novel Deep Derived Feature Fusion (DeepDeFF) Y-shaped sequential deep learning architecture combining raw sequence load data with engineered statistical/cyclical features across 6 recurrent variants (RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU).", font_size=Pt(9.5))
        
        t27 = get_group_shape(g70, "TextBox 27")
        if t27: set_text(t27, "End-to-end Python/PyTorch short-term load forecasting pipeline, trained and validated on 322M+ smart meter readings with sub-4% MAPE accuracy.", font_size=Pt(9.5))
        
        t31 = get_group_shape(g70, "TextBox 31")
        if t31: set_text(t31, "Smart grid distribution system operators (DSOs), microgrid managers, renewable energy aggregators, and utility demand-side management (DSM) systems.", font_size=Pt(9.5))
        
        t35 = get_group_shape(g70, "TextBox 35")
        if t35: set_text(t35, "Stage 5 Completed: Large-scale data preprocessing (322M rows), feature engineering pipeline, and initial DeepDeFF PyTorch model architecture implemented.", font_size=Pt(9.5))
        
        t39 = get_group_shape(g70, "TextBox 39")
        if t39: set_text(t39, "WP2 – Data Preprocessing & Feature Engineering (Completed);\nWP3 – DeepDeFF Model Development & Training (In Progress)", font_size=Pt(9.5))
        
        t43 = get_group_shape(g70, "TextBox 43")
        if t43: set_text(t43, "Approximate project completion based on actual work completed so far: 45 %", font_size=Pt(9.5), bold=True)

    # =========================================================================
    # SLIDE 3: Problem Definition, Need & Motivation (3 / 20)
    # =========================================================================
    print("Updating Slide 3...")
    s3 = prs.slides[2]
    t11 = get_shape(s3, "TextBox 11")
    if t11:
        lines = [
            "What to Present",
            "1. Problem",
            "Accurate energy demand forecasting in modern smart grids is intensely challenging due to high stochasticity, weather dependencies, diverse consumer consumption behaviors, and the rapid integration of distributed renewable energy sources.",
            "2. Existing Solution & Limitation",
            "Standard statistical models (ARIMA) and standalone recurrent neural networks (LSTM/GRU) fail to simultaneously capture fine-grained consumer consumption momentum and multi-horizon periodic patterns without excessive parameters or extreme computational latency.",
            "3. Need for the Project",
            "Accurate, scalable short-term load forecasting (SLF) is critical for day-ahead generation scheduling, spinning reserve optimization, load shedding mitigation, peak shaving, and lowering operational costs in smart grid infrastructures."
        ]
        set_paragraphs(t11, lines, font_size=Pt(9.5))
        for p in t11.text_frame.paragraphs:
            if p.text in ["What to Present", "1. Problem", "2. Existing Solution & Limitation", "3. Need for the Project"]:
                p.font.bold = True

    # Clear placeholder text boxes in evidence area
    t13 = get_shape(s3, "TextBox 13")
    if t13: set_text(t13, "")
    t14 = get_shape(s3, "TextBox 14")
    if t14: set_text(t14, "")
    
    # Add Figure 1 into Rectangle 12 (pos: 6629400, 1417320, 4919040, 4082400)
    s3.shapes.add_picture(fig1_path, 6640000, 1430000, 4890000, 4050000)

    # =========================================================================
    # SLIDE 4: Approved Objectives, Scope & Expected Outcome (4 / 20)
    # =========================================================================
    print("Updating Slide 4...")
    s4 = prs.slides[3]
    # O1
    t20 = get_shape(s4, "TextBox 20")
    if t20: set_text(t20, "Formulate and implement a Deep Derived Feature Fusion (DeepDeFF) Y-shaped sequential architecture for smart grid short-term load forecasting.", font_size=Pt(9.5))
    t22 = get_shape(s4, "TextBox 22")
    if t22: set_text(t22, "A fully operational PyTorch DeepDeFF model supporting 6 sequential cell variants (RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU) with independent feature branches.", font_size=Pt(9.5))
    # O2
    t26 = get_shape(s4, "TextBox 26")
    if t26: set_text(t26, "Collect, clean, and preprocess large-scale public smart grid datasets (PJM Interconnection hourly & SGSC 30-min interval readings).", font_size=Pt(9.5))
    t28 = get_shape(s4, "TextBox 28")
    if t28: set_text(t28, "End-to-end chunked pipeline successfully cleaning 322M+ records, handling missing values/outliers, and exporting compressed Parquet datasets.", font_size=Pt(9.5))
    # O3
    t32 = get_shape(s4, "TextBox 32")
    if t32: set_text(t32, "Engineer multi-resolution temporal, cyclical, and rolling statistical features capturing load momentum and periodicity.", font_size=Pt(9.5))
    t34 = get_shape(s4, "TextBox 34")
    if t34: set_text(t34, "Production of complete 20+ feature matrix including 48-dim time slots, day-of-week, cyclical sin/cos transforms, lag features, and rolling statistical bounds.", font_size=Pt(9.5))
    # O4
    t38 = get_shape(s4, "TextBox 38")
    if t38: set_text(t38, "Train, evaluate, and benchmark the proposed DeepDeFF variants against standard baseline forecasting models.", font_size=Pt(9.5))
    t40 = get_shape(s4, "TextBox 40")
    if t40: set_text(t40, "Quantified forecast performance on test sets evaluated via MAPE, MAE, RMSE, and R², demonstrating superior accuracy and convergence stability.", font_size=Pt(9.5))
    # O5
    t44 = get_shape(s4, "TextBox 44")
    if t44: set_text(t44, "Conduct ablation study and computational efficiency analysis across unidirectional and bidirectional recurrent cells.", font_size=Pt(9.5))
    t46 = get_shape(s4, "TextBox 46")
    if t46: set_text(t46, "Comprehensive comparative benchmark showing percentage error reduction and computational trade-offs among all 6 sequential variants.", font_size=Pt(9.5))
    # Scope & Outcome
    t47 = get_shape(s4, "TextBox 47")
    if t47: set_text(t47, "Scope / Boundaries: Covers short-term electric load forecasting (half-hourly and hourly) utilizing public benchmark smart-grid data (PJM 12 regions and SGSC 322M rows). Includes full data lifecycle: pipeline engineering, feature extraction, PyTorch model training, and comparative evaluation. Scope is software/algorithmic forecasting without direct real-time physical hardware actuator deployment.", font_size=Pt(9.5))
    t48 = get_shape(s4, "TextBox 48")
    if t48: set_text(t48, "Expected Technical Outcome: ■ Simulation Model  □ Prototype  ■ Software/Algorithm  □ Integrated System  ■ Experimental Validation  □ Product/Technology Demonstrator", font_size=Pt(10), bold=True)

    # =========================================================================
    # SLIDE 5: Zeroth Review Recommendations & Action Taken (5 / 20)
    # =========================================================================
    print("Updating Slide 5...")
    s5 = prs.slides[4]
    # Row 1
    t20 = get_shape(s5, "TextBox 20")
    if t20: set_text(t20, "Finalize a suitable, high-volume dataset for developing and benchmarking the smart grid energy forecasting model.", font_size=Pt(9.5))
    t22 = get_shape(s5, "TextBox 22")
    if t22: set_text(t22, "The Smart Grid Smart City (SGSC) 16.5 GB dataset (~322 million 30-min readings) and 12 PJM regional hourly datasets were acquired and benchmarked.", font_size=Pt(9.5))
    t24 = get_shape(s5, "TextBox 24")
    if t24: set_text(t24, "■ Completed   □ Ongoing", font_size=Pt(9.5), bold=True)
    t26 = get_shape(s5, "TextBox 26")
    if t26: set_text(t26, "EEE-PJ-03 Part D & Part E (Dataset Specifications)", font_size=Pt(9.5))
    # Row 2
    t28 = get_shape(s5, "TextBox 28")
    if t28: set_text(t28, "Clearly define the feature engineering strategy and mathematical formulation for time-series inputs.", font_size=Pt(9.5))
    t30 = get_shape(s5, "TextBox 30")
    if t30: set_text(t30, "Formulated the complete 20-field feature matrix including cyclical sin/cos transforms, time-slot one-hot encodings, lag features, and rolling statistical bounds.", font_size=Pt(9.5))
    t32 = get_shape(s5, "TextBox 32")
    if t32: set_text(t32, "■ Completed   □ Ongoing", font_size=Pt(9.5), bold=True)
    t34 = get_shape(s5, "TextBox 34")
    if t34: set_text(t34, "EEE-PJ-03 Part F & Feature Spec Manual", font_size=Pt(9.5))
    # Row 3
    t36 = get_shape(s5, "TextBox 36")
    if t36: set_text(t36, "Address memory and computational limitations when training deep learning models on large-scale smart meter data.", font_size=Pt(9.5))
    t38 = get_shape(s5, "TextBox 38")
    if t38: set_text(t38, "Implemented chunked stream processing (1M rows/chunk), PyArrow Parquet columnar compression (8.8× ratio), and PyTorch DataLoader batch streaming.", font_size=Pt(9.5))
    t40 = get_shape(s5, "TextBox 40")
    if t40: set_text(t40, "■ Completed   □ Ongoing", font_size=Pt(9.5), bold=True)
    t42 = get_shape(s5, "TextBox 42")
    if t42: set_text(t42, "EEE-PJ-03 Part G (Methodology & Implementation)", font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 6: Literature, State-of-the-Art & Research Gap (6 / 20)
    # =========================================================================
    print("Updating Slide 6...")
    s6 = prs.slides[5]
    # Ref 1
    t24 = get_shape(s6, "TextBox 24")
    if t24: set_text(t24, "Wahab et al., 2021, IEEE Access (Vol. 9)", font_size=Pt(9))
    t26 = get_shape(s6, "TextBox 26")
    if t26: set_text(t26, "Deep Derived Feature Fusion (DeepDeFF) with Y-shaped sequential models", font_size=Pt(9))
    t28 = get_shape(s6, "TextBox 28")
    if t28: set_text(t28, "BiLSTM with derived statistical features achieved superior SLF accuracy across diverse international datasets", font_size=Pt(9))
    t30 = get_shape(s6, "TextBox 30")
    if t30: set_text(t30, "Evaluated primarily on small isolated subsets; requires optimization for massive multi-million row smart meter streams", font_size=Pt(9))
    # Ref 2
    t34 = get_shape(s6, "TextBox 34")
    if t34: set_text(t34, "Pentsos et al., 2025, IEEE Transactions on Smart Grid", font_size=Pt(9))
    t36 = get_shape(s6, "TextBox 36")
    if t36: set_text(t36, "Hybrid LSTM-Transformer architecture for multi-series load forecasting", font_size=Pt(9))
    t38 = get_shape(s6, "TextBox 38")
    if t38: set_text(t38, "Multi-head attention captured long-range cross-series dependencies effectively", font_size=Pt(9))
    t40 = get_shape(s6, "TextBox 40")
    if t40: set_text(t40, "Prohibitive computational complexity and GPU memory requirements for practical local grid operations", font_size=Pt(9))
    # Ref 3
    r43 = get_shape(s6, "Rectangle 43")
    if r43: set_text(r43, "Zhang et al., 2026, IEEE Transactions on Smart Grid", font_size=Pt(9))
    t46 = get_shape(s6, "TextBox 46")
    if t46: set_text(t46, "PatchGRU + multi-scale patching and feature interaction", font_size=Pt(9))
    t48 = get_shape(s6, "TextBox 48")
    if t48: set_text(t48, "Efficient long-sequence and multivariate forecasting without severe error accumulation", font_size=Pt(9))
    t50 = get_shape(s6, "TextBox 50")
    if t50: set_text(t50, "Does not decouple raw load dynamics from engineered rolling statistical bounds and tariff cycles", font_size=Pt(9))
    # Ref 4
    t54 = get_shape(s6, "TextBox 54")
    if t54: set_text(t54, "Amalou et al., 2022, Elsevier - Energy Reports", font_size=Pt(9))
    t56 = get_shape(s6, "TextBox 56")
    if t56: set_text(t56, "Comparative analysis of standard RNN, LSTM, and GRU architectures", font_size=Pt(9))
    t58 = get_shape(s6, "TextBox 58")
    if t58: set_text(t58, "Single-branch GRU offered the best trade-off between training speed and basic accuracy", font_size=Pt(9))
    t60 = get_shape(s6, "TextBox 60")
    if t60: set_text(t60, "Standard single-stream inputs fail to capture complex cyclical patterns and rolling volatility", font_size=Pt(9))
    # Gap
    t63 = get_shape(s6, "TextBox 63")
    if t63: set_text(t63, "Existing load forecasting models either rely on standard single-branch RNN/LSTM architectures that struggle to simultaneously learn instantaneous and rolling statistical patterns, or employ hyper-complex Transformers with prohibitive computational overhead. A critical research gap exists for an efficient, modular Y-shaped architecture that fuses raw sequential data with engineered derived features, evaluated rigorously across massive real-world smart grid deployments.", font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 7: Proposed Solution, Novelty & Engineering Contribution (7 / 20)
    # =========================================================================
    print("Updating Slide 7...")
    s7 = prs.slides[6]
    t20 = get_shape(s7, "TextBox 20")
    if t20: set_text(t20, "Standard single-stream LSTM, GRU, or computationally heavy Transformers.", font_size=Pt(9))
    t22 = get_shape(s7, "TextBox 22")
    if t22: set_text(t22, "Deep Derived Feature Fusion (DeepDeFF): Dual-branch Y-shaped sequential network processing raw load & derived statistical features independently before dense fusion.", font_size=Pt(9))
    
    t26 = get_shape(s7, "TextBox 26")
    if t26: set_text(t26, "High prediction error on volatile consumer peaks (MAPE > 6-10%) or excessive GPU hours.", font_size=Pt(9))
    t28 = get_shape(s7, "TextBox 28")
    if t28: set_text(t28, "Target sub-4% MAPE, faster convergence, and lightweight recurrent units (20 hidden units).", font_size=Pt(9))
    
    t32 = get_shape(s7, "TextBox 32")
    if t32: set_text(t32, "Complex multi-million parameter models prone to overfitting and high training cost.", font_size=Pt(9))
    t34 = get_shape(s7, "TextBox 34")
    if t34: set_text(t34, "Highly compact footprint (under 50K parameters), robust to sensor noise via IQR clipping and rolling stats.", font_size=Pt(9))
    
    t38 = get_shape(s7, "TextBox 38")
    if t38: set_text(t38, "Coarse grid-level forecasts insufficient for localized demand response and renewable balancing.", font_size=Pt(9))
    t40 = get_shape(s7, "TextBox 40")
    if t40: set_text(t40, "Granular 30-min consumer forecasting enables accurate peak shaving, distributed renewable integration, and grid stability.", font_size=Pt(9))
    
    t42 = get_shape(s7, "TextBox 42")
    if t42: set_text(t42, "1. Dual-branch Y-shaped DeepDeFF architecture supporting 6 recurrent cell types (RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU).\n2. Domain-driven feature fusion decoupling raw sequence dynamics from rolling statistical bounds.\n3. Highly scalable end-to-end pipeline streaming 322M+ smart meter intervals via Parquet compression.", font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 9: Engineering Design & Key Technical Decisions (Data/Pipeline) (9 / 20)
    # =========================================================================
    print("Updating Slide 9...")
    s9 = prs.slides[8]
    # Row 1
    t71 = get_shape(s9, "TextBox 71")
    if t71: set_text(t71, "Dataset Selection & Scale", font_size=Pt(9), bold=True)
    t72 = get_shape(s9, "TextBox 72")
    if t72: set_text(t72, "SGSC 30-min (16.5 GB, 322M rows) + PJM Hourly (12 regions)", font_size=Pt(9))
    t73 = get_shape(s9, "TextBox 73")
    if t73: set_text(t73, "Combines macro-grid regional load patterns with granular consumer-level interval resolution", font_size=Pt(9))
    t74 = get_shape(s9, "TextBox 74")
    if t74: set_text(t74, "Python / Pandas / PyArrow", font_size=Pt(9))
    
    # Row 2
    t64 = get_shape(s9, "TextBox 64")
    if t64: set_text(t64, "Data Storage & I/O Optimization", font_size=Pt(9), bold=True)
    t66 = get_shape(s9, "TextBox 66")
    if t66: set_text(t66, "Apache Parquet with Snappy Columnar Compression", font_size=Pt(9))
    r23 = get_shape(s9, "Rectangle 23")
    if r23: set_text(r23, "Reduces disk footprint from 37.2 GB to 4.1 GB (8.8× compression) and accelerates I/O read speeds by ~10×", font_size=Pt(9))
    r33 = get_shape(s9, "Rectangle 33")
    if r33: set_text(r33, "PyArrow / FastParquet", font_size=Pt(9))
    
    # Row 3
    t36 = get_shape(s9, "TextBox 36")
    if t36: set_text(t36, "Outlier & Anomaly Treatment", font_size=Pt(9), bold=True)
    t38 = get_shape(s9, "TextBox 38")
    if t38: set_text(t38, "Negative clipping (min 0) + 99.9th percentile upper bound cap (2.929 kWh)", font_size=Pt(9))
    t40 = get_shape(s9, "TextBox 40")
    if t40: set_text(t40, "Eliminates smart meter communication glitches and sensor error spikes without distorting genuine peak demand", font_size=Pt(9))
    t42 = get_shape(s9, "TextBox 42")
    if t42: set_text(t42, "NumPy / Pandas", font_size=Pt(9))

    # Row 4
    t44 = get_shape(s9, "TextBox 44")
    if t44: set_text(t44, "Temporal & Cyclical Feature Representation", font_size=Pt(9), bold=True)
    t46 = get_shape(s9, "TextBox 46")
    if t46: set_text(t46, "Continuous sin/cos transforms (hour, month) + 48 time-slot OHE", font_size=Pt(9))
    t48 = get_shape(s9, "TextBox 48")
    if t48: set_text(t48, "Preserves circular boundary continuity (e.g., 23:30 to 00:00) and distinct diurnal tariff intervals", font_size=Pt(9))
    t50 = get_shape(s9, "TextBox 50")
    if t50: set_text(t50, "Scikit-Learn / NumPy", font_size=Pt(9))

    # Row 5
    t52 = get_shape(s9, "TextBox 52")
    if t52: set_text(t52, "Rolling Statistical & Momentum Features", font_size=Pt(9), bold=True)
    t54 = get_shape(s9, "TextBox 54")
    if t54: set_text(t54, "Lag memories (30m, 1h, 6h, 24h, 7d) & rolling 6h mean/std", font_size=Pt(9))
    t56 = get_shape(s9, "TextBox 56")
    if t56: set_text(t56, "Supplies short-term load momentum and rolling volatility bounds to the derived neural branch", font_size=Pt(9))
    t58 = get_shape(s9, "TextBox 58")
    if t58: set_text(t58, "Pandas / SciPy", font_size=Pt(9))

    # =========================================================================
    # SLIDE 10: Engineering Design - Model Architecture & Hyperparameters
    # =========================================================================
    print("Updating Slide 10...")
    s10 = prs.slides[9]
    t69 = get_shape(s10, "TextBox 69")
    if t69: set_text(t69, "DEEP LEARNING MODEL ARCHITECTURE & HYPERPARAMETER SPECIFICATIONS", font_size=Pt(11), bold=True)
    
    tbl1 = get_shape(s10, "Table 1")
    if tbl1 and tbl1.has_table:
        table = tbl1.table
        table_data = [
            ["Design Decision / Parameter", "Selected Value / Option", "Technical Basis / Calculation", "Tool / Platform"],
            ["Core Model Architecture", "DeepDeFF (Y-Shaped Sequential Fusion)", "Decouples raw sequence dynamics from engineered statistical features, preventing gradient dilution", "PyTorch 2.x"],
            ["Sequential Recurrent Cells", "6 Variants: RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU", "20 hidden units, Dropout=0.2; enables rigorous trade-off study between accuracy and latency", "PyTorch nn.Module"],
            ["Objective Function & Optimization", "Custom MAPE Loss (ε = 1e-7) with Adam Optimizer (lr=0.001)", "Directly minimizes target industry percentage error metric while ensuring zero-division numerical stability", "PyTorch / Python"]
        ]
        for r_idx, row_vals in enumerate(table_data):
            for c_idx, val in enumerate(row_vals):
                cell = table.cell(r_idx, c_idx)
                cell.text = val
                p = cell.text_frame.paragraphs[0]
                p.font.name = "Times New Roman"
                p.font.size = Pt(9.5)
                if r_idx == 0:
                    p.font.bold = True

    # =========================================================================
    # SLIDE 11: Project Timeline / Gantt Chart (10 A / 20)
    # =========================================================================
    print("Updating Slide 11...")
    s11 = prs.slides[10]
    # Update WP labels
    wp_labels = [
        ("TextBox 35", "WP1 – Literature Review & Research Gap"),
        ("TextBox 57", "WP2 – Dataset Acquisition & Preprocessing"),
        ("TextBox 79", "WP3 – Feature Engineering & Matrix Specification"),
        ("TextBox 101", "WP4 – DeepDeFF Model Development (PyTorch)"),
        ("TextBox 123", "WP5 – Model Training & Baseline Benchmarking"),
        ("TextBox 145", "WP6 – Validation & Comparative Performance Evaluation"),
        ("TextBox 167", "WP7 – Conference Paper Manuscript & Publication"),
        ("TextBox 217", "WP8 – Final Integration, Report & Demonstration")
    ]
    for name, lbl in wp_labels:
        sh = get_shape(s11, name)
        if sh: set_text(sh, lbl, font_size=Pt(9.5), bold=True)

    # Fill Gantt cells
    gantt_plan = {
        0: {0: '●', 1: '●'},                         # WP1: Jul-Aug (Completed)
        1: {1: '●', 2: '●'},                         # WP2: Aug-Sep (Completed)
        2: {2: '●', 3: '●'},                         # WP3: Sep-Oct (Completed)
        3: {3: '●', 4: '■', 5: '■'},                 # WP4: Oct (Done), Nov-Dec (Planned)
        4: {5: '■', 6: '■'},                         # WP5: Dec-Jan (Planned)
        5: {6: '■', 7: '■'},                         # WP6: Jan-Feb (Planned)
        6: {7: '■', 8: '■'},                         # WP7: Feb-Mar (Planned)
        7: {8: '■', 9: '■'}                          # WP8: Mar-Apr (Planned)
    }
    month_xs = [2964960, 3778560, 4592520, 5406120, 6220080, 7034040, 7847640, 8661600, 9475200, 10289160]
    wp_ys = [2167920, 2579400, 2990880, 3402360, 3813840, 4225320, 4572360, 4984920]

    for r_idx, wy in enumerate(wp_ys):
        for c_idx, mx in enumerate(month_xs):
            symbol = gantt_plan.get(r_idx, {}).get(c_idx, '')
            best_sh = None
            min_dist = float('inf')
            for sh in s11.shapes:
                if sh.has_text_frame and "TextBox" in sh.name:
                    dist = abs(sh.left - mx) + abs(sh.top - wy)
                    if dist < min_dist:
                        min_dist = dist
                        best_sh = sh
            if min_dist < 200000 and best_sh:
                color = RGBColor(0, 128, 0) if symbol == '●' else RGBColor(30, 80, 180)
                set_text(best_sh, symbol, font_size=Pt(12), bold=True, color=color, align=PP_ALIGN.CENTER)

    # =========================================================================
    # SLIDE 12: Status At First Review (10B / 20)
    # =========================================================================
    print("Updating Slide 12...")
    s12 = prs.slides[11]
    tbl3 = get_shape(s12, "Table 3")
    if tbl3 and tbl3.has_table:
        table = tbl3.table
        col0_lines = [
            "• Comprehensive literature survey on deep learning for STLF (10+ papers analyzed)",
            "• Ingestion and chunked cleaning of 322M+ smart meter intervals (SGSC) & 12 PJM regional datasets",
            "• Engineered 20+ temporal, cyclical, and rolling statistical features",
            "• Columnar Parquet conversion achieving 8.8× storage compression (37.2 GB to 4.1 GB)",
            "• Core DeepDeFF Y-shaped PyTorch architecture developed supporting 6 recurrent variants"
        ]
        col1_lines = [
            "• Training DeepDeFF variants (RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU) on customer subsets",
            "• Optimization of custom MAPE loss function with zero-protection epsilon (1e-7)",
            "• Hyperparameter tuning (hidden units, dropout rates, learning rates)"
        ]
        col2_lines = [
            "• Full comparative evaluation against baseline models (ARIMA, standalone LSTM/GRU, XGBoost)",
            "• Multi-customer cross-validation across diverse consumer consumption profiles",
            "• Preparation of draft conference manuscript for IEEE publication"
        ]
        set_paragraphs(table.cell(1, 0), col0_lines, font_size=Pt(9.5))
        set_paragraphs(table.cell(1, 1), col1_lines, font_size=Pt(9.5))
        set_paragraphs(table.cell(1, 2), col2_lines, font_size=Pt(9.5))

    # =========================================================================
    # SLIDE 13: Work Packages & Evidence Traceability (11 / 20)
    # =========================================================================
    print("Updating Slide 13...")
    s13 = prs.slides[12]
    # Row 1 (WP1)
    t34 = get_shape(s13, "TextBox 34"); set_text(t34, "NA", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t36 = get_shape(s13, "TextBox 36"); set_text(t36, "E1", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t38 = get_shape(s13, "TextBox 38"); set_text(t38, "Completed", font_size=Pt(9.5), align=PP_ALIGN.CENTER, bold=True)
    # Row 2 (WP2)
    t48 = get_shape(s13, "TextBox 48"); set_text(t48, "SPI1", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t50 = get_shape(s13, "TextBox 50"); set_text(t50, "E2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t52 = get_shape(s13, "TextBox 52"); set_text(t52, "Completed", font_size=Pt(9.5), align=PP_ALIGN.CENTER, bold=True)
    # Row 3 (WP3)
    t56 = get_shape(s13, "TextBox 56"); set_text(t56, "Development of DeepDeFF hybrid deep learning-based forecasting model", font_size=Pt(9))
    t58 = get_shape(s13, "TextBox 58"); set_text(t58, "DeepDeFF Y-shaped PyTorch architecture for smart-grid energy forecasting", font_size=Pt(9))
    t62 = get_shape(s13, "TextBox 62"); set_text(t62, "NA", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t64 = get_shape(s13, "TextBox 64"); set_text(t64, "E3", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t66 = get_shape(s13, "TextBox 66"); set_text(t66, "Ongoing", font_size=Pt(9.5), align=PP_ALIGN.CENTER, bold=True)
    # Row 4 (WP4)
    t70 = get_shape(s13, "TextBox 70"); set_text(t70, "Training, hyperparameter tuning and evaluation of the forecasting model", font_size=Pt(9))
    t72 = get_shape(s13, "TextBox 72"); set_text(t72, "Trained forecasting model and performance results using MAE, RMSE and MAPE", font_size=Pt(9))
    t76 = get_shape(s13, "TextBox 76"); set_text(t76, "NA", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t78 = get_shape(s13, "TextBox 78"); set_text(t78, "E4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t80 = get_shape(s13, "TextBox 80"); set_text(t80, "Ongoing", font_size=Pt(9.5), align=PP_ALIGN.CENTER, bold=True)
    # Row 5 (WP5)
    t84 = get_shape(s13, "TextBox 84"); set_text(t84, "Comparison with existing forecasting approaches and ablation study", font_size=Pt(9))
    t86 = get_shape(s13, "TextBox 86"); set_text(t86, "Comparative benchmark tables, ablation analysis, and performance metrics", font_size=Pt(9))
    t90 = get_shape(s13, "TextBox 90"); set_text(t90, "SPI2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t92 = get_shape(s13, "TextBox 92"); set_text(t92, "E4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    t94 = get_shape(s13, "TextBox 94"); set_text(t94, "Not Started", font_size=Pt(9.5), align=PP_ALIGN.CENTER)

    # =========================================================================
    # SLIDE 14: Actual Work Completed After Zeroth Review (12 / 20)
    # =========================================================================
    print("Updating Slide 14...")
    s14 = prs.slides[13]
    # Row 1
    set_text(get_shape(s14, "TextBox 24"), "WP1", font_size=Pt(9.5), bold=True, align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 26"), "Exhaustive survey of 10+ recent IEEE/Elsevier papers on deep learning STLF; identified single-stream limitations", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 28"), "Aajindev P, Karthika S", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 30"), "E1", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 32"), "Literature Synthesis & Problem Formulation", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 34"), "■ Completed   □ Ongoing", font_size=Pt(9), bold=True)
    # Row 2
    set_text(get_shape(s14, "TextBox 36"), "WP2", font_size=Pt(9.5), bold=True, align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 38"), "Chunked processing pipeline for 16.5 GB SGSC dataset; handled negative readings, clipped outliers at 99.9th percentile, converted to Parquet", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 40"), "Vaishnav Venugopal, Dhiya T P", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 42"), "E2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 44"), "Large-Scale Data Ingestion & Storage Optimization", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 46"), "■ Completed   □ Ongoing", font_size=Pt(9), bold=True)
    # Row 3
    set_text(get_shape(s14, "TextBox 48"), "WP2/3", font_size=Pt(9.5), bold=True, align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 50"), "Engineered 48-interval time slots, day-of-week, weekend flags, cyclical sin/cos transforms, lag memory, and rolling 6h/24h statistics", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 52"), "Karthika S, Dhiya T P", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 54"), "E2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 56"), "Feature Matrix & Preprocessing Specification", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 58"), "■ Completed   □ Ongoing", font_size=Pt(9), bold=True)
    # Row 4
    set_text(get_shape(s14, "TextBox 60"), "WP3/4", font_size=Pt(9.5), bold=True, align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 62"), "Implemented DeepDeFF PyTorch architecture with Y-shaped dual recurrent branches and custom MAPE loss function", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 64"), "Vaishnav Venugopal, Aajindev P", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 66"), "E3", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s14, "TextBox 68"), "Deep Learning Model Architecture & Codebase", font_size=Pt(9))
    set_text(get_shape(s14, "TextBox 70"), "■ Completed   □ Ongoing", font_size=Pt(9), bold=True)
    # Status
    t71 = get_shape(s14, "TextBox 71")
    if t71: set_text(t71, "Overall Progress: 45 %     Schedule Status: ■ Ahead   □ On Schedule   □ Slightly Delayed   □ Significantly Delayed", font_size=Pt(10), bold=True)

    # =========================================================================
    # SLIDE 15: Simulation / Modelling / Development Results (13 / 20)
    # =========================================================================
    print("Updating Slide 15...")
    s15 = prs.slides[14]
    # Clear placeholder text boxes
    t12 = get_shape(s15, "TextBox 12"); set_text(t12, "")
    t13 = get_shape(s15, "TextBox 13"); set_text(t13, "")
    # Add Figure 2 into Rectangle 11 (pos: 594360, 1325880, 6629040, 4343040)
    s15.shapes.add_picture(fig2_path, 605000, 1335000, 6605000, 4320000)
    
    # Fill Result Details table on the right
    t21 = get_shape(s15, "TextBox 21"); set_text(t21, "WP2 / WP3", font_size=Pt(9.5), bold=True)
    t25 = get_shape(s15, "TextBox 25"); set_text(t25, "E2 / E3", font_size=Pt(9.5), bold=True)
    t29 = get_shape(s15, "TextBox 29"); set_text(t29, "322M rows SGSC dataset; 30-min interval; Python 3.10 / PyArrow / PyTorch", font_size=Pt(9))
    t33 = get_shape(s15, "TextBox 33"); set_text(t33, "8.8× compression (37.2 GB -> 4.1 GB); Zero data loss; 20 engineered features; Full PyTorch DeepDeFF pipeline functional", font_size=Pt(9))
    t37 = get_shape(s15, "TextBox 37"); set_text(t37, "■ Verified   □ Preliminary   □ Further Validation Required", font_size=Pt(9), bold=True)

    # =========================================================================
    # SLIDE 16: Results, Analysis & Preliminary Validation (14 / 20)
    # =========================================================================
    print("Updating Slide 16...")
    s16 = prs.slides[15]
    # Clear overlapping note
    t78 = get_shape(s16, "TextBox 78"); set_text(t78, "")
    # Row 1
    set_text(get_shape(s16, "TextBox 22"), "Dataset Ingestion & Storage Footprint", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 24"), "37.2 GB (Uncompressed Raw CSV)", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 26"), "< 8.0 GB (Compressed Format)", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 28"), "4.1 GB (PyArrow Parquet)", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 30"), "Target achieved: 89% storage reduction and 10× faster query load time", font_size=Pt(9))
    # Row 2
    set_text(get_shape(s16, "TextBox 32"), "Data Cleaning & Integrity (Missing/Outliers)", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 34"), "Unbounded sensor spikes (>50 kWh) & negative values", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 36"), "Clean, zero-negative, bounded load data", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 38"), "100% negative clipped; 99.9% capped at 2.929 kWh", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 40"), "Target achieved: Clean continuous time series ready for neural net training", font_size=Pt(9))
    # Row 3
    set_text(get_shape(s16, "TextBox 42"), "Feature Space Dimensionality & Coverage", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 44"), "Single timestamp + raw kWh reading (2 features)", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 46"), "Multi-resolution temporal & statistical features (>15)", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 48"), "20 engineered features (cyclical, lag, rolling)", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 50"), "Target achieved: Fully specified in Feature Manual and verified", font_size=Pt(9))
    # Row 4
    set_text(get_shape(s16, "TextBox 52"), "Model Forecast Accuracy (Target MAPE)", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 54"), "Standard LSTM: 6.5% - 8.2% MAPE (Literature)", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 56"), "< 4.5% MAPE on test dataset", font_size=Pt(9))
    set_text(get_shape(s16, "TextBox 58"), "DeepDeFF architecture implemented; initial setup verified", font_size=Pt(9), bold=True)
    set_text(get_shape(s16, "TextBox 60"), "Full batch multi-customer training and hyperparameter tuning in progress", font_size=Pt(9))
    # Bottom notes
    t62 = get_shape(s16, "TextBox 62")
    if t62: set_text(t62, "End-to-end data pipeline, Parquet compression (8.8×), 20-feature extraction matrix, and PyTorch DeepDeFF model architecture.", font_size=Pt(9))
    t64 = get_shape(s16, "TextBox 64")
    if t64: set_text(t64, "Multi-customer cross-validation, 6-variant comparative ablation study (RNN, LSTM, GRU, Bi-variants), and MAPE convergence across all seasons.", font_size=Pt(9))
    t65 = get_shape(s16, "TextBox 65")
    if t65: set_text(t65, "Present Technical Limitation: High memory usage during long-sequence backpropagation; mitigated via mini-batch customer sampling and PyTorch DataLoader streaming.", font_size=Pt(9))

    # =========================================================================
    # SLIDE 17: OBE, SDG & Sustainability Traceability (15 / 20)
    # =========================================================================
    print("Updating Slide 17...")
    s17 = prs.slides[16]
    # Clear overlapping note
    t105 = get_shape(s17, "TextBox 105"); set_text(t105, "")
    # Row 1 (O1)
    set_text(get_shape(s17, "TextBox 31"), "CO2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 33"), "WP3", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 35"), "SDG 7 (Affordable & Clean Energy)", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 37"), "SPI1", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 39"), "Single-branch LSTM", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 41"), "DeepDeFF modular architecture", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 43"), "E3: DeepDeFF PyTorch model implemented with 6 variants", font_size=Pt(9))
    # Row 2 (O2)
    set_text(get_shape(s17, "TextBox 47"), "CO4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 49"), "WP2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 51"), "SDG 9 (Industry & Infrastructure)", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 53"), "SPI2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 55"), "Raw 37GB CSV files", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 57"), "80% compression & clean data", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 59"), "E2: 322M rows cleaned; 8.8× compression in Parquet verified", font_size=Pt(9))
    # Row 3 (O3)
    set_text(get_shape(s17, "TextBox 63"), "CO4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 65"), "WP2", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 67"), "SDG 11 (Sustainable Cities)", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 69"), "NA", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 71"), "No derived features", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 73"), "20+ temporal & statistical features", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 75"), "E2: Feature spec manual & automated extraction complete", font_size=Pt(9))
    # Row 4 (O4)
    set_text(get_shape(s17, "TextBox 79"), "CO3", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 81"), "WP4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 83"), "SDG 13 (Climate Action)", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 85"), "SPI3", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s17, "TextBox 87"), "High peak forecast error (>7%)", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 89"), "MAPE < 4.0% enabling peak shaving", font_size=Pt(9))
    set_text(get_shape(s17, "TextBox 91"), "E4: Model training setup ready; evaluation underway", font_size=Pt(9))

    # =========================================================================
    # SLIDE 18: Publication, IPR & Technology Deliverable Plan (16 / 20)
    # =========================================================================
    print("Updating Slide 18...")
    s18 = prs.slides[17]
    # Clear overlapping note
    t80 = get_shape(s18, "TextBox 80"); set_text(t80, "")
    # Row 1 (Conference Paper)
    set_text(get_shape(s18, "TextBox 22"), "High / In Progress", font_size=Pt(9), bold=True)
    set_text(get_shape(s18, "TextBox 24"), "Performance comparison of DeepDeFF Y-shaped dual-branch sequential models for short-term load forecasting using large-scale smart meter datasets", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 26"), "Complete comparative ablation study across all 6 recurrent variants and draft manuscript for IEEE International Conference", font_size=Pt(9))
    # Row 2 (Journal Paper)
    set_text(get_shape(s18, "TextBox 30"), "Moderate / Phase II", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 32"), "Extended multi-horizon energy forecasting framework incorporating distributed renewable generation and weather covariates", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 34"), "Consolidate seasonal evaluation results and prepare comprehensive benchmark against Transformer architectures", font_size=Pt(9))
    # Row 3 (Patent / Utility)
    set_text(get_shape(s18, "TextBox 38"), "To be Assessed", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 40"), "Specific hardware-efficient pipeline for streaming smart meter interval feature fusion", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 42"), "Review novelty of feature fusion mechanism with project guide and IPR cell", font_size=Pt(9))
    # Row 4 (Design Registration)
    set_text(get_shape(s18, "TextBox 46"), "NA", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 48"), "NA (Software-based algorithmic research)", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 50"), "NA", font_size=Pt(9))
    # Row 5 (Copyright)
    set_text(get_shape(s18, "TextBox 54"), "High / In Progress", font_size=Pt(9), bold=True)
    set_text(get_shape(s18, "TextBox 56"), "Open-source modular Python/PyTorch preprocessing and load forecasting pipeline for smart grid analytics", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 58"), "Structure codebase, document API, and register software copyright", font_size=Pt(9))
    # Row 6 (Prototype / Product)
    set_text(get_shape(s18, "TextBox 62"), "Moderate / In Progress", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 64"), "Software prediction dashboard for smart grid load forecasting", font_size=Pt(9))
    set_text(get_shape(s18, "TextBox 66"), "Develop interactive web interface for trained models", font_size=Pt(9))
    # Primary Target
    t67 = get_shape(s18, "TextBox 67")
    if t67: set_text(t67, "PRIMARY TARGET DELIVERABLE: ■ Conference Paper   □ Journal Paper   □ Patent/IPR   ■ Copyright (Software Pipeline)   □ Prototype/Product", font_size=Pt(9.5), bold=True)

    # =========================================================================
    # SLIDE 19: Challenges, Risks & Corrective Actions (17 / 20)
    # =========================================================================
    print("Updating Slide 19...")
    s19 = prs.slides[18]
    # Clear overlapping note
    t75 = get_shape(s19, "TextBox 75"); set_text(t75, "")
    # Row 1
    set_text(get_shape(s19, "TextBox 22"), "Extremely large dataset volume (16.5 GB CSV, 322M rows) causing RAM exhaustion during processing", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 24"), "Out-of-memory (OOM) crashes during data ingestion and feature calculation", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 26"), "Implemented chunked stream processing (1M rows/chunk) and converted to PyArrow columnar Parquet format", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 28"), "Vaishnav Venugopal, Dhiya T P", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 30"), "Completed (Oct 2026)", font_size=Pt(9), bold=True)
    # Row 2
    set_text(get_shape(s19, "TextBox 32"), "Sensor anomalies, communication dropouts, and negative readings in raw smart meter intervals", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 34"), "Distorted statistical features and exploding gradients during neural network training", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 36"), "Applied strict zero-clipping for negative values and 99.9th percentile IQR capping for upper spikes", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 38"), "Karthika S, Dhiya T P", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 40"), "Completed (Oct 2026)", font_size=Pt(9), bold=True)
    # Row 3
    set_text(get_shape(s19, "TextBox 42"), "Vanishing/exploding gradients and slow convergence across 6 deep recurrent sequential variants", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 44"), "Delayed model training and sub-optimal forecasting accuracy", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 46"), "Implemented gradient clipping, Adam optimizer with adaptive learning rate, and custom MAPE loss with epsilon (1e-7)", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 48"), "Aajindev P, Vaishnav Venugopal", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 50"), "November 2026", font_size=Pt(9))
    # Row 4
    set_text(get_shape(s19, "TextBox 52"), "Semester VII end-semester examinations causing reduced project time in November", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 54"), "Potential schedule delay in multi-customer model training", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 56"), "Completed data preprocessing and model architecture ahead of schedule; automated training scripts to run in background", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 58"), "Entire Project Team", font_size=Pt(9))
    set_text(get_shape(s19, "TextBox 60"), "Mid Nov 2026", font_size=Pt(9))
    # Support
    t61 = get_shape(s19, "TextBox 61")
    if t61: set_text(t61, "Support Required (if any): ■ Guide Technical Input   □ Laboratory   □ Components   ■ Software/License   □ Test Facility   □ Industry/Expert Support   □ None", font_size=Pt(9.5), bold=True)

    # =========================================================================
    # SLIDE 20: Plan Before Next Review (18 / 20)
    # =========================================================================
    print("Updating Slide 20...")
    s20 = prs.slides[19]
    # Clear overlapping note
    t75 = get_shape(s20, "TextBox 75"); set_text(t75, "")
    # Row 1
    set_text(get_shape(s20, "TextBox 22"), "WP4 – Train and fine-tune DeepDeFF variants (RNN, LSTM, GRU, BiRNN, BiLSTM, BiGRU)", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 24"), "Trained PyTorch model checkpoints, training loss curves, and validation MAPE metrics", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 26"), "E3 / E4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s20, "TextBox 28"), "Vaishnav Venugopal, Aajindev P", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 30"), "15 Dec 2026", font_size=Pt(9))
    # Row 2
    set_text(get_shape(s20, "TextBox 32"), "WP5 – Benchmark DeepDeFF against baseline models (ARIMA, XGBoost, Standalone LSTM)", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 34"), "Comprehensive comparative performance table (MAE, RMSE, MAPE, R²)", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 36"), "E4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s20, "TextBox 38"), "Aajindev P, Karthika S", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 40"), "05 Jan 2027", font_size=Pt(9))
    # Row 3
    set_text(get_shape(s20, "TextBox 42"), "WP5 – Conduct ablation study on basic vs derived feature branches", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 44"), "Quantitative report demonstrating percentage accuracy gain provided by derived features", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 46"), "E4", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s20, "TextBox 48"), "Karthika S, Dhiya T P", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 50"), "20 Jan 2027", font_size=Pt(9))
    # Row 4
    set_text(get_shape(s20, "TextBox 52"), "WP7 – Prepare draft conference manuscript based on empirical results", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 54"), "Completed 6-page IEEE conference format draft paper ready for guide review", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 56"), "E5", font_size=Pt(9.5), align=PP_ALIGN.CENTER)
    set_text(get_shape(s20, "TextBox 58"), "Entire Project Team", font_size=Pt(9))
    set_text(get_shape(s20, "TextBox 60"), "10 Feb 2027", font_size=Pt(9))
    # Bottom notes
    t61 = get_shape(s20, "TextBox 61")
    if t61: set_text(t61, "Next Major Milestone: Completion of Multi-Model Benchmark & IEEE Conference Manuscript Draft", font_size=Pt(9.5), bold=True)
    t62 = get_shape(s20, "TextBox 62")
    if t62: set_text(t62, "Primary Research / IPR Deliverable – Next Action: Finalize experimental MAPE comparison tables and submit manuscript to guide for review", font_size=Pt(9.5), bold=True)

    # =========================================================================
    # SLIDE 21: Conclusion / Review Summary (19 / 20)
    # =========================================================================
    print("Updating Slide 21...")
    s21 = prs.slides[20]
    t13 = get_shape(s21, "TextBox 13")
    if t13: set_text(t13, "Highly stochastic and non-linear smart grid electricity demand causes high forecast error in single-stream models.", font_size=Pt(9.5))
    t41 = get_shape(s21, "TextBox 41")
    if t41: set_text(t41, "") # clear duplicate
    t16 = get_shape(s21, "TextBox 16")
    if t16: set_text(t16, "A dual-branch Deep Derived Feature Fusion (DeepDeFF) sequential architecture combining raw sequence dynamics with rolling statistical momentum.", font_size=Pt(9.5))
    t19 = get_shape(s21, "TextBox 19")
    if t19: set_text(t19, "End-to-end preprocessing of 322M+ smart meter intervals, 8.8× Parquet compression, 20-feature matrix, and PyTorch DeepDeFF model implementation.", font_size=Pt(9.5))
    t22 = get_shape(s21, "TextBox 22")
    if t22: set_text(t22, "Zero data-loss pipeline reducing storage from 37.2 GB to 4.1 GB and functional Y-shaped architecture supporting 6 recurrent variants.", font_size=Pt(9.5))
    t25 = get_shape(s21, "TextBox 25")
    if t25: set_text(t25, "IEEE Conference paper on DeepDeFF load forecasting and open-source smart grid preprocessing tool.", font_size=Pt(9.5))
    t28 = get_shape(s21, "TextBox 28")
    if t28: set_text(t28, "Multi-customer model training, hyperparameter optimization, and baseline benchmarking (ARIMA, XGBoost, LSTM).", font_size=Pt(9.5))
    t39 = get_shape(s21, "TextBox 39")
    if t39: set_text(t39, "") # clear dot

    # =========================================================================
    # SLIDE 22: References (20 / 20)
    # =========================================================================
    print("Updating Slide 22...")
    s22 = prs.slides[21]
    t12 = get_shape(s22, "TextBox 12")
    if t12:
        refs = [
            "[1] A. Wahab, M. A. Tahir, N. Iqbal, A. Ul-Hasan, F. Shafait, and S. M. R. Kazmi, “A Novel Technique for Short-Term Load Forecasting Using Sequential Models and Feature Engineering,” IEEE Access, vol. 9, pp. 96221–96232, 2021, doi: 10.1109/ACCESS.2021.3093481.",
            "[2] V. Pentsos, G. T. Andreou, and A. S. Bouhouras, “A Hybrid LSTM-Transformer Model for Power Load Forecasting,” IEEE Transactions on Smart Grid, vol. 16, no. 3, pp. 2624–2634, 2025.",
            "[3] Z. Zhang et al., “A Multi-Task End-to-End Multivariate Long-Sequence Time Series Prediction Model for Load Forecasting,” IEEE Transactions on Smart Grid, vol. 17, no. 1, pp. 715–732, 2026.",
            "[4] I. Amalou, N. Mouhni, and A. Abdali, “Multivariate Time Series Prediction by RNN Architectures for Energy Consumption Forecasting,” Energy Reports, vol. 8, pp. 1084–1091, 2022, doi: 10.1016/j.egyr.2022.07.139.",
            "[5] M. A. Majeed, S. Phichaisawat, F. Asghar, and U. Hussan, “Data-Driven Optimized Load Forecasting: An LSTM-Based RNN Approach for Smart Grids,” IEEE Access, vol. 13, 2025, doi: 10.1109/ACCESS.2025.3576303.",
            "[6] S. B. Melhem et al., “EdgeAI-Powered Hybrid ESN-GRU Model for High-Accuracy and Efficient Short-Term Load Forecasting in Smart Grids,” IEEE Access, vol. 13, 2025, doi: 10.1109/ACCESS.2025.3631672.",
            "[7] C. Briggs, Z. Fan, and P. Andras, “Federated Learning for Short-Term Residential Load Forecasting,” IEEE Open Journal of the Power and Energy Society, vol. 9, pp. 174–185, 2022, doi: 10.1109/OAJPE.2022.3206220.",
            "[8] D. Syed et al., “Inductive Transfer and Deep Neural Network Learning-Based Cross-Model Method for Short-Term Load Forecasting in Smart Grids,” IEEE Canadian Journal of Electrical and Computer Engineering, vol. 46, no. 2, pp. 157–168, 2023, doi: 10.1109/ICJECE.2023.3253547.",
            "[9] Z. Masood, R. Gantassi, and Y. Choi, “Enhancing Short-Term Electric Load Forecasting for Households Using Quantile LSTM and Clustering-Based Probabilistic Approach,” IEEE Access, vol. 12, pp. 88120–88135, 2024, doi: 10.1109/ACCESS.2024.3406439."
        ]
        set_paragraphs(t12, refs, font_size=Pt(8.5))

    # =========================================================================
    # SLIDE 23: Thank You (20 / 20)
    # =========================================================================
    print("Updating Slide 23...")
    s23 = prs.slides[22]
    t2 = get_shape(s23, "TextBox 2")
    if t2:
        thank_you_lines = [
            "ENERGY DEMAND FORECASTING OF SMART GRID USING DEEP LEARNING",
            "Group No: ER27A15 | Department of Electrical & Electronics Engineering | TKMCE, Kollam",
            "",
            "Thank you! We welcome questions, comments, and valuable suggestions from the review panel."
        ]
        set_paragraphs(t2, thank_you_lines, font_size=Pt(12), align=PP_ALIGN.CENTER)
        if len(t2.text_frame.paragraphs) >= 1:
            t2.text_frame.paragraphs[0].font.bold = True
            t2.text_frame.paragraphs[0].font.size = Pt(14)

    prs.save(output_path)
    print(f"Presentation successfully saved to: {output_path}")

if __name__ == "__main__":
    update_presentation()
