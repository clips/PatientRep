#-*- coding: utf-8 -*-

'''
@author madhumita
'''

import sys
import os
import xml.etree.ElementTree as ET

class Db_config(object):
    
    def __init__(self, root):    
        for db_prop in root.findall('db'):
            self.MYSQL_HOST = db_prop.find('MYSQL_HOST').text
            self.MYSQL_UNAME = db_prop.find('MYSQL_UNAME').text
            self.MYSQL_PASS = db_prop.find('MYSQL_PASS').text
            self.MYSQL_PORT = int(db_prop.find('MYSQL_PORT').text)
            self.MYSQL_MIMIC_DB = db_prop.find('MYSQL_MIMIC_DB').text
            self.MYSQL_TABLE_DEM = db_prop.find('MYSQL_TABLE_DEM').text
            self.MYSQL_TABLE_HOSP = db_prop.find('MYSQL_TABLE_HOSP').text
            self.MYSQL_TABLE_ICD9D = db_prop.find('MYSQL_TABLE_ICD9D').text
            self.MYSQL_TABLE_LETTER = db_prop.find('MYSQL_TABLE_LETTER').text
            self.MYSQL_TABLE_ORREPORT = db_prop.find('MYSQL_TABLE_ORREPORT').text
            self.MYSQL_TABLE_PROTOCOL =  db_prop.find('MYSQL_TABLE_PROTOCOL').text
            self.MYSQL_TABLE_NOTES = db_prop.find('MYSQL_TABLE_NOTES').text
            self.MYSQL_TABLE_OTHERUNSTR = db_prop.find('MYSQL_TABLE_OTHERUNSTR').text
            self.MYSQL_TABLE_DIAGCAT = db_prop.find('MYSQL_TABLE_DIAGCAT').text
            self.MYSQL_TABLE_DRG = db_prop.find('MYSQL_TABLE_DRG').text
            
            self.DB_EXT = db_prop.find('DB_EXT').text
            self.TABLE_MOLIS = db_prop.find('TABLE_MOLIS').text

class Path_config(object):
    def __init__(self, root):
        for path_prop in root.findall('paths'):
            self.PATH_INPUT = path_prop.find('PATH_INPUT').text
            self.PATH_RESOURCES = path_prop.find('PATH_RESOURCES').text 
            self.PATH_OUTPUT = path_prop.find('PATH_OUTPUT').text
            self.PATH_CONTENT_MIMIC = path_prop.find('PATH_CONTENT_MIMIC').text
            self.PATH_EMBEDDING = path_prop.find('PATH_EMBEDDING').text
            self.PATH_PROCESSED_UCTO_MIMIC = path_prop.find('PATH_PROCESSED_UCTO_MIMIC').text
            self.PATH_PROCESSED_CLAMP_MIMIC = path_prop.find('PATH_PROCESSED_CLAMP_MIMIC').text
            self.PATH_NORMALIZED_MIMIC = path_prop.find('PATH_NORMALIZED_MIMIC').text
            self.PATH_CONTENT_CONCEPTS_MIMIC = path_prop.find('PATH_CONTENT_CONCEPTS_MIMIC').text
            self.PATH_CONTENT_CUI_MIMIC = path_prop.find('PATH_CONTENT_CUI_MIMIC').text
            self.PATH_CUI_MIMIC = path_prop.find('PATH_CUI_MIMIC').text
        self._update_if_frozen()
        self._create_path_dirs()
    
    def _update_if_frozen(self):
        '''
        Update the path variables if running in a pyinstaller bundle
        '''
        if getattr( sys, 'frozen', False ):
            # running in a bundle
            self.PATH_INPUT = '../../'+self.PATH_INPUT
            self.PATH_OUTPUT = '../../'+self.PATH_OUTPUT
            self.PATH_RESOURCES = '../../'+self.PATH_RESOURCES
#     
    def _create_path_dirs(self):
        '''
        Create the directories for path variables if they do not exist
        '''
        self._create_dir(self.PATH_INPUT)
        self._create_dir(self.PATH_RESOURCES)
        self._create_dir(self.PATH_OUTPUT)
        
    def _create_dir(self,dirName):
        '''
        Create a directory with a given path
        '''
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    
    
        
class Res_config(object):
    def __init__(self, cfg_file):
        self.escape_chars = {
                        'á':'a',
                        'â':'a',
                        'æ':'ae',
                        'à':'a',
                        'å':'a',
                        'ã':'a',
                        'ä':'a',
                        'ë':'e', 
                        'è':'e', 
                        'é':'e', 
                        'ê':'e', 
                        'ç':'c',
                        'í':'i',
                        'î':'i',
                        'ì':'i',
                        'ï':'i',
                        'ñ':'n',
                        'ó':'o',
                        'ô':'o',
                        'ò':'o',
                        'ø':'o',
                        'õ':'o',
                        'ö':'o',
                        'œ':'oe',
                        'ú':'u',
                        'û':'u',
                        'ù':'u',
                        'ü':'u',
                        'ÿ':'y'
                        }


class Run_config(object):
    def __init__(self, root):
        for run_prop in root.findall('run'):
            self.seed = int(run_prop.find('seed').text)
            self.lang = run_prop.find('lang').text.upper()
            self.init_data = (run_prop.find('init_data').text in ['True', 'true'])
            self.processed = (run_prop.find('processed').text in ['True', 'true'])
            self.p_type = run_prop.find('pipeline').text
            self.pipeline = run_prop.find('path_pipeline').text
            
            self.split_data = (run_prop.find('split_data').text in ['True', 'true'])
            self.featurize = (run_prop.find('featurize').text in ['True', 'true'])
            self.feat_type = run_prop.find('feat_type').text.lower()
            self.content_type = run_prop.find('content_type').text.lower()
            self.val_type = run_prop.find('val_type').text.lower()
            self.feat_level = run_prop.find('feat_level').text.lower()
            self.resample = (run_prop.find('resample').text in ['True', 'true'])
            
            for feat_selection in run_prop.findall('feat_selection'):
                self.select_best_feats = (feat_selection.find('select_best').text in ['True', 'true'])
                self.n_feats = float(feat_selection.find('n').text)
                self.n_feats_type = feat_selection.find('type').text.lower()
                self.sel_criteria = feat_selection.find('sel_criteria').text.lower()
                
            for pretraining in run_prop.findall('pretraining'):
                self.pretrain = pretraining.find('pretrain').text in ['True', 'true']
                self.rep_type = pretraining.find('rep_type').text.lower()
                self.load_model = pretraining.find('load_model').text.lower() == 'true'
                
                if self.rep_type == 'sdae':
                    for sdae in pretraining.findall('sdae'):
                        self.n_hid_layers = int(sdae.find('n_hid_layers').text)
                        self.n_epochs = int(sdae.find('n_epochs').text)
                        self.n_hid = [int(i) for i in sdae.find('n_hid').text.split(',')]
                        self.dropout = [float(i) for i in sdae.find('dropout').text.split(',')]
            
            for ensemble_tech in run_prop.findall('ensemble_tech'):
                self.ensemble = ensemble_tech.find('ensemble').text.lower() == 'true'
                self.ens_models = ensemble_tech.find('models').text.lower().split(',')
                self.ens_type = ensemble_tech.find('type').text.lower()
                        
            for ffnn in run_prop.findall('ffnn'):
                self.n_hid_layers_ffnn = int(ffnn.find('n_hid_layers').text)
                self.n_epochs_ffnn = int(ffnn.find('n_epochs').text)
                self.act_fn_ffnn = ffnn.find('act_fn').text
                if self.n_hid_layers_ffnn > 0:
                    self.n_hid_ffnn = [int(i) for i in ffnn.find('n_hid').text.split(',')]
                else:
                    self.n_hid_ffnn = list()
                             
            for hnn in run_prop.findall('hnn'):
                self.max_sent_len = hnn.find('max_sent_len').text.lower() == 'true'
                self.max_note_len = hnn.find('max_note_len').text.lower() == 'true'
                self.max_pt_len = hnn.find('max_pt_len').text.lower() == 'true'
                
                self.sent_emb_dim = int(hnn.find('sent_emb_dim').text)
                self.note_emb_dim = int(hnn.find('note_emb_dim').text)
                self.pt_emb_dim = int(hnn.find('pt_emb_dim').text)
                        
            for embedding in run_prop.findall('embedding'):
                self.process_embeddings = embedding.find('process_embeddings').text.lower()
                self.emb_dim = int(embedding.find('dim').text)
                self.emb_algo = embedding.find('algo').text.lower()
                self.emb_tool = embedding.find('tool').text.lower()
            
            self.pred_algo = run_prop.find('pred_algo').text.lower()
            self.pred_type = run_prop.find('pred_type').text.lower()
class Config(object):
    def __init__(self,cfg_file):
        
        with open(cfg_file, encoding='UTF-8') as f:
            root = ET.parse(f).getroot()
            
            self.db_cfg = Db_config(root)
            self.path_cfg = Path_config(root)
            self.res_cfg = Res_config(root)
            self.run_cfg = Run_config(root)
