def _filepair_chooser(Filenames):
    
    # syntax to call this definition: 
    # processing_filepairs = _processing_filepair(Filenames)
    
    filepair = []   
    
    # matching filepairs for SIMI and GS data (including sync file), normal paced walk
    if  len(Filenames['Filepair_Normal_v1']) > len(Filenames['Filepair_Normal_v2']):
        filepair.append(Filenames['Filepair_Normal_v1'])
    else:
        filepair.append(Filenames['Filepair_Normal_v2'])
    
    # matching filepairs for SIMI and GS data (including sync file), fast paced walk    
    if  len(Filenames['Filepair_Fast_v1']) > len(Filenames['Filepair_Fast_v2']):
        filepair.append(Filenames['Filepair_Fast_v1'])
    else:
        filepair.append(Filenames['Filepair_Fast_v2'])  

    # matching filepairs for SIMI and GS data (including sync file), slow paced walk
    if  len(Filenames['Filepair_Slow_v1']) > len(Filenames['Filepair_Slow_v2']):
        filepair.append(Filenames['Filepair_Slow_v1'])
    else:
        filepair.append(Filenames['Filepair_Slow_v2'])

    # matching filepairs for SIMI and GS data (including sync file), carpeted surface walk
    if  len(Filenames['Filepair_Carpet_v1']) > len(Filenames['Filepair_Carpet_v2']):
        filepair.append(Filenames['Filepair_Carpet_v1'])
    else:
        filepair.append(Filenames['Filepair_Carpet_v2'])

    return filepair


if 'Filenames' in locals() or 'Filenames' in globals():
  print('Filenames dictionary exists') 
  
elif 'Filenames' not in locals() or 'Filenames' not in globals():
  print('Filenames dictionary does not exist. Make sure to run "SIMIpy_m_filenames.py" ')
  
