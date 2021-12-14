# SIMIpy_m
SIMI Motion, compared to GAITRite System Metrics


    # 1) "SIMIpy_m_main.py" to process all trials for one patricipant.
        # Required: The folder structure should contain all 4 trial types collected for that participant.
                # Each trial type should contain (3) files: one .txt file for SIMI data, and two .csv files for GS data ans GS sync:
                # filepath = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_SIMIMCS1_Normal_Walk_marker_processed_7_13_2021.txt"
                # filepath_GS = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS.csv"
                # filepath_GS_sync = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS_sync.csv"
        # Calculates all SIMI vs. GS metrics and generates comparisons based on processed SIMI and GS data.            
        # Generates metrics for all trials and saves into a "Batch_Outputs" dictionary.
  
    # 2) "SIMIpy_m_GUI.py" to select any one of the four trial types above, for further processing or visualization.

    # 3) "SIMIpy_m_metric.py" to process the metrics for the selected individual trial.
          
    # 4) "SIMIpy_m_plots.py" to plot the results such as GS events, and calculated metrics based on FVA, HMA, Heel to Heel Distance Algorithms.

    # Scripts required:
          # SIMIpy_m_Event_Metrics.py
          # SIMIpy_m_filenames.py
          # SIMIpy_m_GUI.py
          # SIMIpy_m_main.py
          # SIMIpy_m_metrics.py
          # SIMIpy_m_metrics_SIMI_passes.py
          # SIMIpy_m_PKMAS_sync.py
          # SIMIpy_m_plots.py
          # SIMIpy_m_processing_filepair.py
  
