# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

keys = {"HeelStrike_SIMI_Normal": [],
        "HeelStrike_SIMI_Fast": [],
        "HeelStrike_SIMI_Slow": [],
        "HeelStrike_SIMI_Carpet": [],

        "ToeOff_SIMI_Normal": [],
        "ToeOff_SIMI_Fast": [],
        "ToeOff_SIMI_Slow": [],
        "ToeOff_SIMI_Carpet": [],

        "HeelStrike_GS_Normal": [],
        "HeelStrike_GS_Fast": [],
        "HeelStrike_GS_Slow": [],
        "HeelStrike_GS_Carpet": [],

        "ToeOff_GS_Normal": [],
        "ToeOff_GS_Fast": [],
        "ToeOff_GS_Slow": [],
        "ToeOff_GS_Carpet": []
        }

Participants_HS_TO = {
    "PN10010002": keys,
    "PN10010003": keys,
    "PN10010004": keys,
    "PN10010005": keys,
    "PN10010006": keys,
    "PN10010007": keys,
    "PN10010008": keys,
    "PN10010009": keys,
    "PN10010010": keys,
    "PN10010011": keys,
    "PN10010012": keys,
    "PN10010013": keys,
    "PN10010014": keys,
    "PN10010015": keys,
    "PN10010016": keys,
    "PN10010017": keys,
    "PN10010018": keys,
    "PN10010019": keys,
    "PN10010020": keys
}


z = "PN" + Filenames['participant_num']

Participants_HS_TO[z]["HeelStrike_SIMI_Normal"] = HeelStrike_SIMI['Normal']
Participants_HS_TO[z]["HeelStrike_SIMI_Fast"] = HeelStrike_SIMI['Fast']
Participants_HS_TO[z]["HeelStrike_SIMI_Slow"] = HeelStrike_SIMI['Slow']
Participants_HS_TO[z]["HeelStrike_SIMI_Carpet"] = HeelStrike_SIMI['Carpet']

Participants_HS_TO[z]["ToeOff_SIMI_Normal"] = ToeOff_SIMI['Normal']
Participants_HS_TO[z]["ToeOff_SIMI_Fast"] = ToeOff_SIMI['Fast']
Participants_HS_TO[z]["ToeOff_SIMI_Slow"] = ToeOff_SIMI['Slow']
Participants_HS_TO[z]["ToeOff_SIMI_Carpet"] = ToeOff_SIMI['Carpet']

Participants_HS_TO[z]["HeelStrike_GS_Normal"] = HeelStrike_GS['Normal']
Participants_HS_TO[z]["HeelStrike_GS_Fast"] = HeelStrike_GS['Fast']
Participants_HS_TO[z]["HeelStrike_GS_Slow"] = HeelStrike_GS['Slow']
Participants_HS_TO[z]["HeelStrike_GS_Fast"] = HeelStrike_GS['Carpet']

Participants_HS_TO[z]["ToeOff_GS_Normal"] = ToeOff_GS['Normal']
Participants_HS_TO[z]["ToeOff_GS_Fast"] = ToeOff_GS['Fast']
Participants_HS_TO[z]["ToeOff_GS_Slow"] = ToeOff_GS['Slow']
Participants_HS_TO[z]["ToeOff_GS_Carpet"] = ToeOff_GS['Carpet']

del z
