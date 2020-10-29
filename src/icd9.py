class ICD9:
    
    def __init__(self):
        
        self.diag_mapping = { '1': 'Infectious and parasitic diseases',
                        '2': 'Neoplasms',
                        '3': 'Endocrine, nutritional and metabolic diseases, and immunity disorders',
                        '4': 'Diseases of the blood and blood-forming organs',
                        '5': 'Mental disorders',
                        '6': 'Diseases of the nervous system',
                        '7': 'Diseases of the sense organs',
                        '8': 'Diseases of the circulatory system',
                        '9': 'Diseases of the respiratory system',
                        '10': 'Diseases of the digestive system',
                        '11': 'Diseases of the genitourinary system',
                        '12': 'Complications of pregnancy, childbirth, and puerperium',
                        '13': 'Diseases of the skin and subcutaneous tissue',
                        '14': 'Diseases of the musculoskeletal system and connective tissue',
                        '15': 'Congenital anomalies',
                        '16': 'Certain conditions originating in the perinatal period',
                        '17': 'Symptoms, signs, and ill-defined conditions',
                        '18': 'Injury and poisoning',
                        'E':'External causes of injury', 
                        'V':'Supplementary classification of factors influencing health status and contact with health services'
                       
                       }
        
        self.diag_classes = { 'Infectious and parasitic diseases' : 0,
                        'Neoplasms' : 1,
                        'Endocrine, nutritional and metabolic diseases, and immunity disorders' : 2,
                        'Diseases of the blood and blood-forming organs' : 3,
                        'Mental disorders': 4,
                        'Diseases of the nervous system' : 5,
                        'Diseases of the sense organs' : 6,
                        'Diseases of the circulatory system' : 7,
                        'Diseases of the respiratory system' : 8,
                        'Diseases of the digestive system' : 9,
                        'Diseases of the genitourinary system': 10,
                        'Complications of pregnancy, childbirth, and puerperium' : 11,
                        'Diseases of the skin and subcutaneous tissue' : 12,
                        'Diseases of the musculoskeletal system and connective tissue' : 13,
                        'Congenital anomalies' : 14,
                        'Certain conditions originating in the perinatal period' : 15,
                        'Symptoms, signs, and ill-defined conditions': 16,
                        'Injury and poisoning' : 17,
                        'External causes of injury' : 18, 
                        'Supplementary classification of factors influencing health status and contact with health services' : 19
                       
                       }
        
        def _get_icd9d_cat(self, icd9d):
            
            if icd9d is None:
                return
            
            icd9d = icd9d.upper()
            
            if icd9d[0] == 'V':
                return 'V'
            elif icd9d[0] == 'E':
                return 'E'
            else:
                icd9d = str(icd9d)[0:3]
                if icd9d >= '001' and icd9d <= '139':
                    return '1'
                elif icd9d >= '140' and icd9d <= '239':
                    return '2'
                elif icd9d >= '240' and icd9d <= '279':
                    return '3'
                elif icd9d >= '280' and icd9d <= '289':
                    return '4'
                elif icd9d >= '290' and icd9d <= '319':
                    return '5'
                elif icd9d >= '320' and icd9d <= '359':
                    return '6'
                elif icd9d >= '360' and icd9d <= '389':
                    return '7'
                elif icd9d >= '390' and icd9d <= '459':
                    return '8'
                elif icd9d >= '460' and icd9d <= '519':
                    return '9'
                elif icd9d >= '520' and icd9d <= '579':
                    return '10'
                elif icd9d >= '580' and icd9d <= '629':
                    return '11'
                elif icd9d >= '630' and icd9d <= '679':
                    return '12'
                elif icd9d >= '680' and icd9d <= '709':
                    return '13'
                elif icd9d >= '710' and icd9d <= '739':
                    return '14'
                elif icd9d >= '740' and icd9d <= '759':
                    return '15'
                elif icd9d >= '760' and icd9d <= '779':
                    return '16'
                elif icd9d >= '780' and icd9d <= '799':
                    return '17'
                elif icd9d >= '800' and icd9d <= '999':
                    return '18'
        
        self.proc_mapping = { '1': 'Procedures And Interventions , Not Elsewhere Classified',
                            '2': 'Operations On The Nervous System',
                            '3': 'Operations On The Endocrine System',
                            '4': 'Operations On The Eye',
                            '5': 'Other Miscellaneous Diagnostic And Therapeutic Procedures',
                            '6': 'Operations On The Ear',
                            '7': 'Operations On The Nose, Mouth, And Pharynx',
                            '8': 'Operations On The Respiratory System',
                            '9': 'Operations On The Cardiovascular System',
                            '10': 'Operations On The Hemic And Lymphatic System',
                            '11': 'Operations On The Digestive System',
                            '12': 'Operations On The Urinary System',
                            '13': 'Operations On The Male Genital Organs',
                            '14': 'Operations On The Female Genital Organs',
                            '15': 'Obstetrical Procedures',
                            '16': 'Operations On The Musculoskeletal System',
                            '17': 'Operations On The Integumentary System',
                            '18':'Miscellaneous Diagnostic And Therapeutic Procedures'
                            
                           }
        
        
        
        self.proc_classes = {'Procedures And Interventions , Not Elsewhere Classified': 0,
                            'Operations On The Nervous System' : 1,
                            'Operations On The Endocrine System' : 2,
                            'Operations On The Eye' : 3,
                            'Other Miscellaneous Diagnostic And Therapeutic Procedures' : 4,
                            'Operations On The Ear' : 5,
                            'Operations On The Nose, Mouth, And Pharynx' : 6,
                            'Operations On The Respiratory System' : 7,
                            'Operations On The Cardiovascular System' : 8,
                            'Operations On The Hemic And Lymphatic System' : 9,
                            'Operations On The Digestive System' : 10,
                            'Operations On The Urinary System' : 11,
                            'Operations On The Male Genital Organs' : 12,
                            'Operations On The Female Genital Organs' : 13,
                            'Obstetrical Procedures' : 14,
                            'Operations On The Musculoskeletal System' : 15,
                            'Operations On The Integumentary System' : 16,
                            'Miscellaneous Diagnostic And Therapeutic Procedures' : 17 
                            
                           }

    
                
    def _get_icd9p_cat(self, icd9p):
            
            if icd9p is None:
                return 
            
            icd9p = icd9p.upper()
            icd9p = str(icd9p)[0:2]
            
            if icd9p == '00':
                return '1'
            elif icd9p >= '01' and icd9p <= '05':
                return '2'
            elif icd9p >= '06' and icd9p <= '07':
                return '3'
            elif icd9p >= '08' and icd9p <= '16':
                return '4'
            elif icd9p >= '17' and icd9p <= '17':
                return '5'
            elif icd9p >= '18' and icd9p <= '20':
                return '6'
            elif icd9p >= '21' and icd9p <= '29':
                return '7'
            elif icd9p >= '30' and icd9p <= '34':
                return '8'
            elif icd9p >= '35' and icd9p <= '39':
                return '9'
            elif icd9p >= '40' and icd9p <= '41':
                return '10'
            elif icd9p >= '42' and icd9p <= '54':
                return '11'
            elif icd9p >= '55' and icd9p <= '59':
                return '12'
            elif icd9p >= '60' and icd9p <= '64':
                return '13'
            elif icd9p >= '65' and icd9p <= '71':
                return '14'
            elif icd9p >= '72' and icd9p <= '75':
                return '15'
            elif icd9p >= '76' and icd9p <= '84':
                return '16'
            elif icd9p >= '85' and icd9p <= '86':
                return '17'
            elif icd9p >= '87' and icd9p <= '99':
                return '18'