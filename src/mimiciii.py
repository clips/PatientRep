'''
@author madhumita
'''
import numpy as np
from scipy.sparse.base import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

import random

import sys
import gzip

from sdae import StackedDenoisingAE as SDAE
from model import Model
import utils
import model_utils
import featurize
from dense_rep import DenseRep
import ffnn
import nn_utils
import visualize
from icd9 import ICD9
from clustering import Cluster
from model_interpretation import get_ffnn_input_significance, get_sdae_input_significance
import param_config_utils

class Note(object):
    '''
    Class with details of every note for a patient
    '''
    def __init__(self, noteid, pid, hadmid, rowid, chartdate, category, cat_desc, text):
        '''
        @param noteid: the ID of the note for the patient. Range [0, n-1], where n is the number of notes for the patient. The earliest note has the lowest ID
        @param pid: patient ID
        @param hadmid: hospital admission ID
        @param chartdate: the date when a given note was charted
        @param category: the category of the given note
        @param cat_desc: the description of the note category
        @param text: the content of the given note
        '''
        self.noteid = noteid
        self.pid = pid
        self.hadmid = hadmid
        self.rowid = rowid
        self.chartdate = chartdate
        self.category = category
        self.cat_desc = cat_desc
        self.text = text
                   
class PatientStay(object):
    '''
    Class containing the details of a single stay/ICU admission event of a patient
    '''
    def __init__(self, pid, hadmid, notes, mortality_hosp, mortality_30_days, mortality_1_year):  # , mortality_class):
        '''
        Initial parameters for the class
        @param pid: subject (patient) ID
        @param hadmid: ID of a given stay (hospital admission) of a given patient
        @param notes: list of class Note which contains the details of all the notes of a patient in ascending order of chartdate
        @param mortality_hosp: 1 if patient died within the hospital, 0 otherwise.
        @param mortality_30_days: 1 if patient died within 30 days of hospital admission, 0 otherwise.
        @param mortality_1_year: 1 if patient died within 1 year of hospital admission, 0 otherwise.
#         @param mortality_class (none/hosp/30days/1year): Indicate in which of the considered mortality categories does the patient fall in
        '''
        self.pid = pid
        self.hadmid = hadmid
        self.notes = notes
        self.mortality_hosp = mortality_hosp
        self.mortality_30_days = mortality_30_days
        self.mortality_1_year = mortality_1_year
#         self.mortality_class = mortality_class
        

class MIMICIII(object):
    '''
    Class containing information about all the patient stays from the MIMIC III database.
    It is currently limited to the set of patient stays that have an associated text note, and mortality information.
    '''
    
    def __init__(self, cfg):
        '''
        Finds all the adult patients (>=18) in the MIMICIII dataset with a single hospital admission.
        Finds all the notes for the patients charted before they died or were discharged. Excludes all the discharge notes.
        Only includes the patients with at least one such associated note.
        @param cfg: object of Config class, with different config params
        '''
        
        self.note_cats = set()  # contains all categories that a note can have
        self.patients = list()  # set of objects of class PatientStay
        
        self.train_idx = list()
        self.val_idx = list()
        self.test_idx = list()
        
        self.feats = None  # object of type featurize.Feature
        self.dense_feats = None  # object of type dense_rep.DenseRep
        
        self._get_note_cats(cfg)  # get all the note categories.
        self._find_patients(cfg)  # finds all relevant patient
        self._shuffle_patients(cfg)  # shuffles the patient list to remove ordering bias
        self._get_train_val_test_pt_indices(cfg)  # splits data into train, val and test
        self._reorganize_train_idx()
        
        self.get_mortality_stats()
        
        # serializes the data to files
        self._write_note_cats(cfg)
        self._write_note_texts(cfg)
        self._write_mortality(cfg)
        
        self.serialize(cfg, fname='mimiciii')
        
    def _get_note_cats(self, cfg):
        '''
        Gets all distinct categories that a note can have.
        @param cfg: Config object
        '''
        db_conn = utils.connect_to_db(cfg.db_cfg.MYSQL_HOST, cfg.db_cfg.MYSQL_PORT, cfg.db_cfg.MYSQL_UNAME, cfg.db_cfg.MYSQL_PASS, cfg.db_cfg.MYSQL_MIMIC_DB)
        cur = db_conn.cursor()
        
        cur.execute("select distinct category from " + cfg.db_cfg.MYSQL_MIMIC_DB + ".NOTEEVENTS")
        
        for cat, in cur.fetchall():
            self.note_cats.add(cat)
        db_conn.close()
        
    def _find_patients(self, cfg):
        '''
        Finds all the adult patients (>=18) in the MIMICIII dataset with a single hospital admission. 
        Gets all the notes for the patients charted before they died or were discharged. Excludes all the discharge notes. 
        Also gets the mortality information about the patient.
        @param cfg: Config object
        '''
        total_notes = 0
        
        db_conn = utils.connect_to_db(cfg.db_cfg.MYSQL_HOST, cfg.db_cfg.MYSQL_PORT, cfg.db_cfg.MYSQL_UNAME, cfg.db_cfg.MYSQL_PASS, cfg.db_cfg.MYSQL_MIMIC_DB)
        cur = db_conn.cursor()
        
        db_name = cfg.db_cfg.MYSQL_MIMIC_DB
        
        # Query to get subject (patient ID), hospital admission ID and age of all adults (>=18) with a single hospital admission.
        # Note: age of patients >= 89 is set as 210
        query = '''select a.subject_id, a.hadm_id,  a.hospital_expire_flag as hosp_expire, 
        round( datediff(b.dod, a.dischtime), 2) as death_days, 
        round( DATEDIFF(a.admittime, b.dob) / 365.242, 2) as age 
        from  ''' + db_name + '''.ADMISSIONS a left join ''' + db_name + '''.PATIENTS b on a.SUBJECT_ID = b.SUBJECT_ID 
        where a.SUBJECT_ID in 
        (select subject_id from ''' + db_name + '''.ADMISSIONS group by SUBJECT_ID having count(hadm_id) = 1) having age >= 18'''
        
        cur.execute(query)
        
        for (pid, hadm_id, hosp_death, death_days, age) in cur.fetchall():
            mortality_hosp, mortality_30_days, mortality_1_year, mortality_class = self._get_mortality(hosp_death, death_days)
            (notes, n_notes) = self._get_notes(db_conn, db_name, pid, hadm_id)
            
            total_notes += n_notes
            
            if n_notes:
                cur_pt = PatientStay(pid, hadm_id, notes, mortality_hosp, mortality_30_days, mortality_1_year, mortality_class)
                self.patients.append(cur_pt)

        db_conn.close()
        
        print("Total num of patients: ", len(self.patients))
        print("Total num of notes: ", total_notes)
        
    def _get_mortality(self, hosp_death, death_days):
        '''
        Gets the mortality of a patient - whether the patient died in the hospital, died within 30 days after discharge, or within 1 year after discharge.
        @param hosp_death: 1 if the patient died during the hospital admission (Can be ICU or non ICU death during that admission)
        @param death_days: The number of days after discharge that the patient died
        @return (mortality_hosp, mortality_30_days, mortality_1_year), 
            1 indicates True class, and 0 indicates False class for the first 3 cases, 
#             in mortality_class, 0 indicates None, 1 indicates death within hospital, 2: death within 30 days, and 3: death within 1 year of discharge.
        '''
        mortality_hosp = 0
        mortality_30_days = 0
        mortality_1_year = 0
#         mortality_class = 'none'
        
        if hosp_death:
            mortality_hosp = 1
#             mortality_class = 'hosp'
        
        elif death_days and death_days <= 30:
            mortality_30_days = 1
#             mortality_class  ='30days'
        
        elif death_days and death_days > 30 and death_days <= 365:
            mortality_1_year = 1
#             mortality_class = '1year'
        
        return (mortality_hosp, mortality_30_days, mortality_1_year)  # , mortality_class)
    
    def _get_notes(self, db_conn, db_name, pid, hadmid):
        '''
        Gets all the notes charted for a patient before the discharge or death time, excluding discharge notes (duplicate notes removed).
        @param db_conn: Existing connection to database
        @param db_name: Name of the database containing the MIMIC III data.
        @param pid: Patient ID
        @param hadmid: Hospital admission (stay) ID - present if the patient was admitted to ICU during the stay
        @return (notes  n_notes): set of all Note objects for the patient; n_notes: the num of notes retrieved for the patient, under the given constraints.                     
        '''
        notes = list()
        
        cur = db_conn.cursor()
        
        query = """
        SELECT a.row_id, convert(a.chartdate, datetime) as chartdate, a.category, a.description, a.text
        FROM """ + db_name + """.NOTEEVENTS a join """ + db_name + """.ADMISSIONS b on a.hadm_id = b.hadm_id 
        where a.subject_id = %s and a.HADM_ID = %s and a.iserror is NULL
        and a.category not like %s
        and b.dischtime >= chartdate
        and (b.deathtime is NULL or (b.deathtime >= chartdate))
        order by chartdate asc
        """
        
#         query = '''
#         SELECT a.category, a.text 
#         FROM MIMICIII.NOTEEVENTS
#         where a.subject_id = %s and a.HADM_ID = %s and iserror is NULL 
#         and category not like '%discharge%' 
#         '''
        i = 0
        cur.execute(query, (pid, hadmid, '%discharge%'))
        for (rowid, chartdate, category, cat_desc, text) in cur.fetchall():
            duplicate = False
            for prev_note in notes:
                if prev_note.text == text and prev_note.chardate == chartdate and prev_note.category == category and prev_note.cat_desc == cat_desc:
                    duplicate = True
#                     print("Duplicate note found for pid: ",pid, " and hadmid, ",hadmid)
#                     print(prev_note.text)
#                     print(text)
                    break
            if not duplicate:
                notes.append(Note(i, pid, hadmid, rowid, chartdate, category, cat_desc, text))  # .replace("\n", " ")))
                i += 1
            
#         if len(notes) == 0:
#             print("No notes found for the patient with patient ID: ", pid, " and HADM_ID: ", hadmid)
        
        return (notes, len(notes))
    
    def _get_csv_lables(self, cfg):
        """
        Store gender, marital status, 
        admission location, 
        diagnosis, diagnostic category, 
        procedure category, 
        current_care_unit, 
        number of different medication, and number of total medication
        for a patient
        @param cfg: Config object
        """
        db_conn = utils.connect_to_db(cfg.db_cfg.MYSQL_HOST, cfg.db_cfg.MYSQL_PORT, cfg.db_cfg.MYSQL_UNAME, cfg.db_cfg.MYSQL_PASS, cfg.db_cfg.MYSQL_MIMIC_DB)
        cur = db_conn.cursor()
        
        db_name = cfg.db_cfg.MYSQL_MIMIC_DB
        
        y = self._gen_multiclass_y_indices()
        
        for i, cur_pt in enumerate(self.patients):
            
            if i and not i%100:
                print("Processed ", i ," of ", len(self.patients), " patients") 
                
            pid = cur_pt.pid
            hadmid = cur_pt.hadmid
            
            cur_pt.gender = self._get_gender(cur, db_name, pid)
            cur_pt.mar_stat, self.patients[i].adm_loc, self.patients[i].diagnosis = self._get_adm_loc_diag(cur, db_name, pid, hadmid)
            cur_pt.diag_cats = self._get_diag_cats(cur, db_name, pid, hadmid)
            cur_pt.proc_cats = self._get_proc_cats(cur, db_name, pid, hadmid)
            cur_pt.ccu = self._get_ccu(cur, db_name, pid, hadmid)
            cur_pt.n_med, self.patients[i].n_med_type = self._get_med(cur, db_name, pid, hadmid)
            cur_pt.mortality = y[i]
            
#             print(cur_pt.gender)
#             print(cur_pt.mar_stat)
#             print(cur_pt.adm_loc)
#             print(cur_pt.diagnosis)
#             print(cur_pt.diag_cats)
#             print(cur_pt.proc_cats)
#             print(cur_pt.ccu)
#             print(cur_pt.n_med)
#             print(cur_pt.n_med_type)
#             
        db_conn.close()
        
        self.serialize(cfg, fname='mimiciii_with_labels')
        
    def _get_gender(self, cur, db_name, pid):
        """
        Execute query to get patient gender
        @param cur: database cursor
        @param db_name: name of database
        @param pid: patient ID for the patient whose gender we want to fetch
        @return gender
        """        
        query = "select gender from " + db_name + ".PATIENTS where subject_id = %s"
        cur.execute(query, (pid,))
            
        for gender, in cur.fetchall():
            return gender
    
    def _get_adm_loc_diag(self, cur, db_name, pid, hadmid):
        """
        Get the marital status of the patient at the time of his hospitalization,
        the location where the patient was first admitted to the hospital, 
        and the primary diagnosis associated with patient stay
        @param cur: database cursor
        @param db_name: name of database
        @param pid: patient ID
        @param hadmid: hospital stay/admission ID
        @return (marital_status, admission_location, diagnosis)
        """        
        query = "select marital_status, admission_location, diagnosis from " + db_name + ".ADMISSIONS where subject_id = %s and hadm_id = %s"
        cur.execute(query, (pid, hadmid,))
            
        for (mar_stat, adm_loc, diag) in cur.fetchall():
            return (mar_stat, adm_loc, diag)
        
    def _get_diag_cats(self, cur, db_name, pid, hadmid):
        """
        Get the highest level category of ICD9 diagnostic codes for the patient
        @param cur: database connection cursor
        @param db_name: database name
        @param pid: patient ID
        @param hadmi: hospital admission ID of the patient
        @return set of all ICD9 diagnostic categories associated with the patient
        """
        icd9_inst = ICD9()
        
        icd_cats = list()
        query = "select icd9_code from " + db_name + ".DIAGNOSES_ICD where subject_id = %s and hadm_id = %s order by seq_num asc"
        cur.execute(query, (pid, hadmid,))
            
        for icd9d, in cur.fetchall():
            icd9d_cat = icd9_inst._get_icd9d_cat(icd9d)
            if icd9d_cat is not None:
                icd_cats.append(icd9_inst.diag_mapping[icd9d_cat])
            
        return icd_cats
        
    def _get_proc_cats(self, cur, db_name, pid, hadmid):
        """
        Get the highest level category of ICD9 procedural codes for the patient
        @param cur: database connection cursor
        @param db_name: database name
        @param pid: patient ID
        @param hadmi: hospital admission ID of the patient
        @return set of all ICD9 procedural categories associated with the patient
        """
        
        icd9_inst = ICD9()
        proc_cats = list()
        query = "select icd9_code from " + db_name + ".PROCEDURES_ICD where subject_id = %s and hadm_id = %s order by seq_num asc"
        cur.execute(query, (pid, hadmid,))
            
        for icd9p, in cur.fetchall():
            icd9p_cat = icd9_inst._get_icd9p_cat(icd9p)
            if icd9p_cat is not None:
                proc_cats.append(icd9_inst.proc_mapping[icd9p_cat])
            
        return proc_cats
        
    def _get_ccu(self, cur, db_name, pid, hadmid):
        """
        Get the current_care_unit of the patient just before discharge
        @param cur: database cursor
        @param db_name: name of database
        @param pid: patient ID
        @param hadmid: hospital stay/admission ID
        @return ccu
        """        
        query = "select curr_careunit from " + db_name + ".CALLOUT where subject_id = %s and hadm_id = %s"
        cur.execute(query, (pid, hadmid,))
            
        for ccu, in cur.fetchall():
            return ccu
    
    def _get_med(self, cur, db_name, pid, hadmid):
        """
        Get the total number of medications, and the number of different medication prescribed to a patient
        @param cur: database cursor
        @param db_name: name of database
        @param pid: patient ID
        @param hadmid: hospital stay/admission ID
        @return (num of meds, num of different meds)
        """
        query = "SELECT count(drug), count(distinct drug) FROM " + db_name + ".PRESCRIPTIONS where subject_id = %s and hadm_id = %s"
        
        cur.execute(query, (pid, hadmid,))
            
        for (n_med, n_med_type) in cur.fetchall():
            return (n_med, n_med_type)

    def _shuffle_patients(self, cfg):
        '''
        Shuffles the patient list to remove ordering bias
        @param cfg: config file
        '''
        random.seed(cfg.run_cfg.seed)
        random.shuffle(self.patients)
    
    def _reorganize_train_idx(self):
        """
        Sorts the train index based on the num of notes in a patient, so that patients with comparable number of notes are grouped together in a batch.
        """
        
        n_notes = dict()  # dictionary of {train_pt_idx:n_notes}
        for cur_idx in self.train_idx:
            n_notes[cur_idx] = len(self.patients[cur_idx].notes)
        
        self.train_idx = sorted(n_notes, key=n_notes.get)
                
    def get_mortality_stats(self):
        '''
        Prints the fraction of each type of mortality within patients
        '''
        mortality_h = 0
        mortality_30 = 0
        mortality_1 = 0
        
        total_pt = len(self.patients)
        
        for cur_pt in self.patients:
            if cur_pt.mortality_hosp:
                mortality_h += 1
            elif cur_pt.mortality_30_days:
                mortality_30 += 1
            elif cur_pt.mortality_1_year:
                mortality_1 += 1
                
                
        print("Percentage of patients who died within the hospital: ", float(mortality_h) / total_pt * 100)
        print("Percentage of patients who died within 30 days of discharge: ", float(mortality_30) / total_pt * 100)
        print("Percentage of patients who died within 1 year of discharge: ", float(mortality_1) / total_pt * 100)
        
    def _write_note_texts(self, cfg):
        '''
        Write all the note texts to a file, with the 'pid_hadmid_noteid' as the file name.
        @param cfg: Config object
        '''
        for cur_pt in self.patients:
            for cur_note in cur_pt.notes:
                utils.write_txt_file(cfg.path_cfg.PATH_CONTENT_MIMIC, str(cur_note.pid) + '_' + str(cur_note.hadmid) + '_' + str(cur_note.rowid) + '.txt', cur_note.text)
        
    def _write_note_cats(self, cfg):
        '''
        Writes a json file with all the note categories (set) that occur in the mimic dataset
        @param cfg: Config object
        '''
        utils.write_list(self.note_cats, out_dir=cfg.path_cfg.PATH_INPUT, fname='mimiciii_note_cats.txt')
    
    def _write_mortality(self, cfg):
        '''
        Write labels for patient mortality in JSON format for dictionary {pid_hadmid:(mortality_hosp, mortality_30_days, mortality_1_year)}
        @param cfg: Config object
        '''
        mortality = dict()
        for cur_pt in self.patients:
            mortality[str(cur_pt.pid) + '_' + str(cur_pt.hadmid)] = (cur_pt.mortality_hosp, cur_pt.mortality_30_days, cur_pt.mortality_1_year)
            
        utils.write_data(mortality, out_dir=cfg.path_cfg.PATH_INPUT, f_name='mimiciii_mortality_labels.json')
        
    def serialize(self, cfg, fname):  
        '''
        Serialize the MIMICIII object
        @param cfg: Config file
        '''
        utils.write_data(self, out_dir=cfg.path_cfg.PATH_INPUT, f_name=fname + '.pkl.gz', pickle_data=True, compress=True)
    
    def load_train_val_test_pt_indices(self, cfg):
        '''
        Load train, val, test indices from external file
        '''
        self.train_idx = utils.read_int_list(cfg.path_cfg.PATH_INPUT, fname='mimic_train_idx.txt')
        self.val_idx = utils.read_int_list(cfg.path_cfg.PATH_INPUT, fname='mimic_val_idx.txt')
        self.test_idx = utils.read_int_list(cfg.path_cfg.PATH_INPUT, fname='mimic_test_idx.txt')
         
        self.train_idx = [int(i) for i in self.train_idx]
        self.val_idx = [int(i) for i in self.val_idx]
        self.test_idx = [int(i) for i in self.test_idx]
                    
    def _get_train_val_test_pt_indices(self, cfg):
        '''
        Get training, validation and test patient indices after stratified train/val/test split 
        (class imbalance is preserved)
        @param cfg: config file
        '''
        
        y = self._gen_multiclass_y_indices()
        
        self.train_idx, self.val_idx, self.test_idx = model_utils.get_stratified_train_val_test_split(y, cfg.run_cfg.seed)
        
        self._write_train_val_test_idx(cfg)
        
    def _gen_multiclass_y_indices(self):
        '''
        Generates the codified y (label) matrices for dataset to get train test split
        y values are the labels:
            1 if the patient died in the hospital
            2 if the patient died after 30 days of discharge
            3 if the patient died within 1 year of discharge.
            0 if none of the above labels is true
        @return y
        '''
            
        y = list()
                
        for cur_pt in self.patients:
            if cur_pt.mortality_hosp:
                y.append(1)
            elif cur_pt.mortality_30_days:
                y.append(2)
            elif cur_pt.mortality_1_year:
                y.append(3)
            else:
                y.append(0)
                
        return y        
                
    def _write_train_val_test_idx(self, cfg):
        '''
        Write indices of training, validation, and test instances, corresponding to the patients list
        '''
        utils.write_list(self.train_idx, cfg.path_cfg.PATH_INPUT, fname='mimic_train_idx.txt')
        utils.write_list(self.val_idx, cfg.path_cfg.PATH_INPUT, fname='mimic_val_idx.txt')
        utils.write_list(self.test_idx, cfg.path_cfg.PATH_INPUT, fname='mimic_test_idx.txt')
        
    def featurize(self, cfg):
        '''
        Generate train, val and test features for input data
        @param cfg: Config object
        '''
        self.feats = featurize.Features(dataset=self, cfg=cfg, featurize=cfg.run_cfg.featurize, feat_type=cfg.run_cfg.feat_type, val_type=cfg.run_cfg.val_type)
        
#         self.write_datasets(cfg)
        
    def load_feats_labels(self, cfg, params = None):
        '''
        Load a feature set that has already been written to a pickle file
        @param cfg: Config object
        @param params: dense feature parameters if required
        '''
        np.random.seed(1337)
        self.feats = featurize.Features(dataset=self, cfg=cfg, featurize=cfg.run_cfg.featurize, feat_type=cfg.run_cfg.feat_type, val_type=cfg.run_cfg.val_type)
        
        if cfg.run_cfg.pretrain:
            if cfg.run_cfg.ensemble:
                
                if 'bag' in cfg.run_cfg.ens_models:
                    bag_train = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_train.pkl.gz', pickle_data=True, compress=True).todense()
                    bag_val = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_val.pkl.gz', pickle_data=True, compress=True).todense()
                    bag_test = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_test.pkl.gz', pickle_data=True, compress=True).todense()
             
                if 'doc2vec' in cfg.run_cfg.ens_models:
                    doc2vec_train  = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'doc2vec_dense_train_1.pkl.gz', pickle_data= True, compress = True)
                    doc2vec_val = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'doc2vec_dense_val_1.pkl.gz', pickle_data= True, compress = True) 
                    doc2vec_test  = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'doc2vec_dense_test_1.pkl.gz', pickle_data= True, compress = True) 
                
                if 'sdae' in cfg.run_cfg.ens_models:
                    sdae_train = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_train_1.pkl.gz', pickle_data= True, compress = True)
                    sdae_val = utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_val_1.pkl.gz', pickle_data= True, compress = True)
                    sdae_test =  utils.read_data(cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_test_1.pkl.gz', pickle_data= True, compress = True)
                
                if cfg.run_cfg.ens_type == 'concat':
                    if 'bag' in cfg.run_cfg.ens_models and 'doc2vec' in cfg.run_cfg.ens_models and 'sdae' in cfg.run_cfg.ens_models:
                        self.feats.feats_train = np.concatenate((bag_train, doc2vec_train, sdae_train), axis=1)
                        self.feats.feats_val = np.concatenate((bag_val, doc2vec_val, sdae_val), axis=1)
                        self.feats.feats_test = np.concatenate((bag_test, doc2vec_test, sdae_test), axis=1)
                    elif 'bag' in cfg.run_cfg.ens_models and 'doc2vec' in cfg.run_cfg.ens_models:
                        self.feats.feats_train = np.concatenate((bag_train, doc2vec_train), axis=1)
                        self.feats.feats_val = np.concatenate((bag_val, doc2vec_val), axis=1)
                        self.feats.feats_test = np.concatenate((bag_test, doc2vec_test), axis=1)
                    elif 'bag' in cfg.run_cfg.ens_models and 'sdae' in cfg.run_cfg.ens_models:
                        self.feats.feats_train = np.concatenate((bag_train, sdae_train), axis=1)
                        self.feats.feats_val = np.concatenate((bag_val, sdae_val), axis=1)
                        self.feats.feats_test = np.concatenate((bag_test, sdae_test), axis=1)
                    elif 'sdae' in cfg.run_cfg.ens_models and 'doc2vec' in cfg.run_cfg.ens_models:
                        self.feats.feats_train = np.concatenate((doc2vec_train, sdae_train), axis=1)
                        self.feats.feats_val = np.concatenate((doc2vec_val, sdae_val), axis=1)
                        self.feats.feats_test = np.concatenate((doc2vec_test, sdae_test), axis=1)
            else:
                self.get_dense_rep(cfg, params)
                if cfg.run_cfg.pred_algo.lower() != 'sdae':
                    self.feats.feats_train = self.dense_feats.dense_train
                    self.feats.feats_val = self.dense_feats.dense_val
                    self.feats.feats_test = self.dense_feats.dense_test
        
        self._get_n_classes(cfg)
        self.get_labels(cfg)
        #model_utils.get_label_stats_dense(self.y_train+self.y_val+self.y_test, np.arange(self.n_classes))
        
        if cfg.run_cfg.resample:
            self.feats.feats_train, self.y_train = model_utils.resample(self.feats.feats_train, self.y_train)
            
    def load_dense_feats(self, cfg, params = None):
        """
        Load dense patient representation
        """
        self.dense_feats = DenseRep(cfg, dataset=self, load_model=True, params = params)
            
    def select_best_feats(self, cfg):
        '''
        Replace the current features with the top features based on TF-IDF (log) score
        @param cfg: Config object
        '''
        self.load_feats_labels(cfg)
        self.feats.feats_train, self.feats.feats_val, self.feats.feats_test = self.feats.select_feats(
                                                                                                        self.feats.feats_train,
                                                                                                        self.feats.feats_val,
                                                                                                        self.feats.feats_test,
                                                                                                        self.y_train,
                                                                                                        cfg.run_cfg.n_feats,
                                                                                                        cfg.run_cfg.n_feats_type,
                                                                                                        select_best_overall=cfg.run_cfg.select_best_overall,
                                                                                                        criteria=cfg.run_cfg.sel_criteria,
                                                                                                        val_type=cfg.run_cfg.val_type
                                                                                                        )
#          self.feats.get_feats_from_indices()
        
    def get_multiclass_mortality(self, feat_level):
        """
        return train, val and test labels for multiclass classification setup:
        0 indicates none, 
        1 indicates that the patient died in the hospital, 
        2 indicates that the patient died within 30 days of discharge, 
        3 indicates that the patient died within 1 year of discharge
        """
        y = self._gen_multiclass_y_indices()
        
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(y[idx])
        else:
            self.y_train = [y[idx] for idx in self.train_idx]
        self.y_val = [y[idx] for idx in self.val_idx]
        self.y_test = [y[idx] for idx in self.test_idx]
            
    def get_in_hosp_mortality(self, feat_level):
        """
        return train, val and test labels for binary classification setup: whether the patient died in the hospital.
        0 indicates that the patient did not die in the hospital,
        1 indicates that the patient died in the hospital
        """
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(self.patients[idx].mortality_hosp)
        else:
            self.y_train = [self.patients[idx].mortality_hosp for idx in self.train_idx]
        self.y_val = [self.patients[idx].mortality_hosp for idx in self.val_idx]
        self.y_test = [self.patients[idx].mortality_hosp for idx in self.test_idx]
            
    def get_30_days_mortality(self, feat_level):
        """
        return train, val and test labels for binary classification setup: whether the patient died within 30 days of discharge.
        0 indicates that the patient did not die within 30 days of discharge,
        1 indicates that the patient died within 30 days of discharge
        """
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(self.patients[idx].mortality_30_days)
        else:
            self.y_train = [self.patients[idx].mortality_30_days for idx in self.train_idx]
        self.y_val = [self.patients[idx].mortality_30_days for idx in self.val_idx]
        self.y_test = [self.patients[idx].mortality_30_days for idx in self.test_idx]
            
    def get_1_year_mortality(self, feat_level):
        """
        return train, val and test labels for binary classification setup: whether the patient died within 1 year of discharge.
        0 indicates that the patient did not die within 1 year of discharge,
        1 indicates that the patient died within 1 year of discharge
        """            
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days)
        else:
            self.y_train = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days for idx in self.train_idx]
        self.y_val = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days for idx in self.val_idx]
        self.y_test = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days for idx in self.test_idx]
        
    def get_mortality(self, feat_level):
        """
        return train, val, test labels for binary classification setup: whether the patient dies in the duration of admit to the hospital to 1 year of discharge
        """
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days or self.patients[idx].mortality_hosp)
        else:
            self.y_train = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days or self.patients[idx].mortality_hosp for idx in self.train_idx]
        self.y_val = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days or self.patients[idx].mortality_hosp for idx in self.val_idx]
        self.y_test = [self.patients[idx].mortality_1_year or self.patients[idx].mortality_30_days or self.patients[idx].mortality_hosp for idx in self.test_idx]
    
        
    def get_gender(self, feat_level):
        """
        return train, val, test labels for binary classification setup: patient gender male/female. 0 represents female, 1 represents male
        """
        if feat_level == 'note':
            self.y_train = []
            for idx in self.train_idx:
                for i in range(len(self.patients[idx].notes)):
                    self.y_train.append(self.patients[idx].gender.upper() == 'M')
        else:
            self.y_train = [int(self.patients[idx].gender.upper() == 'M') for idx in self.train_idx]
        self.y_val = [int(self.patients[idx].gender.upper() == 'M') for idx in self.val_idx]
        self.y_test = [int(self.patients[idx].gender.upper() == 'M') for idx in self.test_idx]
        
    def get_pri_diag_cat(self):
        """
        return train, val, test labels for binary classification setup: most relevant diagnostic category for patient.
        Only for pt level analysis
        """
        
        self.y_train, self.feats.feats_train = self._update_data_on_class(self.train_idx, self.feats.feats_train, f_type = 'pri_diag_cat')
        self.y_val, self.feats.feats_val = self._update_data_on_class(self.val_idx, self.feats.feats_val, f_type = 'pri_diag_cat')
        self.y_test, self.feats.feats_test = self._update_data_on_class(self.test_idx, self.feats.feats_test, f_type = 'pri_diag_cat')
    
    def get_pri_proc_cat(self):
        """
        return train, val, test labels for binary classification setup: most relevant procedural category for patient.
        Only for pt level analysis, not note level
        """
        
        self.y_train, self.feats.feats_train = self._update_data_on_class(self.train_idx, self.feats.feats_train, f_type = 'pri_proc_cat')
        self.y_val, self.feats.feats_val = self._update_data_on_class(self.val_idx, self.feats.feats_val, f_type = 'pri_proc_cat')
        self.y_test, self.feats.feats_test = self._update_data_on_class(self.test_idx, self.feats.feats_test, f_type = 'pri_proc_cat')
           
    def _update_data_on_class(self, pt_idx, feats, f_type):
        
        cur_icd9 = ICD9()
        
        y = list()
        to_delete = list()
        
        for i, idx in enumerate(pt_idx): 
            if f_type == 'pri_diag_cat':
                if len(self.patients[idx].diag_cats) > 0:
                    y.append(cur_icd9.diag_classes[self.patients[idx].diag_cats[0]])
                else:
#                     print("Deleting patient %s because no diagnostic category found" %(self.patients[idx].pid))
                    to_delete.append(i)
            elif f_type == 'pri_proc_cat':
                if len(self.patients[idx].proc_cats) > 0:
                    y.append(cur_icd9.proc_classes[self.patients[idx].proc_cats[0]])
                else:
#                     print("Deleting patient %s because no procedural category found" %(self.patients[idx].pid))
                    to_delete.append(i)
            
        feats = self._delete_inst(feats, to_delete)
        return y, feats
               
    def _delete_inst(self, feats, inst_row):
        """
        Delete the given rowns from a matrix
        @param feats: sparse or dense feature matrix
        @param inst_row: list of row IDs to delete
        """
        rows = np.arange(feats.shape[0])
        if issparse(feats):
            rows = np.delete(rows, inst_row, axis = 0)
            feats = csr_matrix(csc_matrix(feats)[rows,:]) 
        else:
            feats = np.delete(feats, inst_row, axis = 0)
        
        return feats
                
    def get_dense_rep(self, cfg, params):
        '''
        Create pretrained dense representation of input data
        @param cfg: Config object
        @param n_classes: number of output classes
        @param params: configuration parameters for generating dense representation
        '''
        self.dense_feats = DenseRep(cfg, self, load_model=cfg.run_cfg.load_model, params = params)
        
    def gen_embeddings(self, cfg):
        '''
        Generate word embeddings
        @param cfg: Config object
        '''
        self.embeddings.gen_embeddings(dim=cfg.run_cfg.emb_dim, dir_in=cfg.path_cfg.PATH_EMBEDDING, f_in='mimic_all_train_tokenized.txt', dir_model=cfg.path_cfg.PATH_EMBEDDING, f_model='mimic', algo=cfg.run_cfg.emb_algo, tool=cfg.run_cfg.emb_tool)
    
    def load_embeddings(self, cfg):
        '''
        Load pretrained word embeddings models
        @param cfg: Config object
        '''
        self.embeddings.load_embedding_model(dir_model=cfg.path_cfg.PATH_EMBEDDING, f_model='mimic', dim=cfg.run_cfg.emb_dim, algo=cfg.run_cfg.emb_algo, tool=cfg.run_cfg.emb_tool)
    
    def _get_n_classes(self, cfg):
        """
        Get number of classes for the given task
        """
        if cfg.run_cfg.pred_type == 'multiclass':
            self.n_classes = 4
        elif cfg.run_cfg.pred_type == 'in_hosp':
            self.n_classes = 2
        elif cfg.run_cfg.pred_type == '30_days':
            self.n_classes = 2
        elif cfg.run_cfg.pred_type == '1_year':
            self.n_classes = 2
        elif cfg.run_cfg.pred_type == 'mortality':
            self.n_classes = 2
        elif cfg.run_cfg.pred_type == 'gender':
            self.n_classes = 2
        elif cfg.run_cfg.pred_type == 'pri_diag_cat':
            self.n_classes = 20
        elif cfg.run_cfg.pred_type == 'pri_proc_cat':
            self.n_classes = 18
        else:
            self.n_classes = 0
            print("Please enter correct prediction type")
            
    def get_labels(self, cfg):
        """
        Get labels and number of classes based on current classification setup
        """
        if cfg.run_cfg.pred_type == 'multiclass':
            print("getting multiclass mortality labels")
            self.get_multiclass_mortality(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == 'in_hosp':
            print("getting in_hosp mortality labels")
            self.get_in_hosp_mortality(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == '30_days':
            print("getting 30 days post discharge mortality labels")
            self.get_30_days_mortality(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == '1_year':
            print("getting labels for patient mortality within 1 year after discharge")
            self.get_1_year_mortality(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == 'mortality':
            print("getting labels for patient mortality within 1 year from admission")
            self.get_mortality(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == 'gender':
            print("Getting gender labels")
            self.get_gender(cfg.run_cfg.feat_level)
        elif cfg.run_cfg.pred_type == 'pri_diag_cat':
            print("Getting primary diagnostic category labels")
            self.get_pri_diag_cat()
        elif cfg.run_cfg.pred_type == 'pri_proc_cat':
            print("Getting primary procedural category labels")
            self.get_pri_proc_cat()
        elif cfg.run_cfg.pred_type == 'input_recon':
            print("Loading original input features as labels:")
            self.y_train = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_train.pkl.gz', pickle_data=True, compress=True)
            self.y_val = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_val.pkl.gz', pickle_data=True, compress=True)
            self.y_test = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_test.pkl.gz', pickle_data=True, compress=True)
            self.n_classes = self.y_train.shape[1]
        else:
            print("Please enter correct prediction type")
            print("You have currently entered: ", cfg.run_cfg.pred_type)
        
    def train_classifier(self, cfg, params = None):
        """
        Trains a classifier for the given classification task and algorithm
        @return trained classifier
        """
        np.random.seed(1337)
        fit_model = self._fit(cfg, params)
        
        return fit_model
        
    def _fit(self, cfg, params):
        """
        Fit the required classifier to the data according to given parameter setting and return trained model
        """
        
        if cfg.run_cfg.pretrain and cfg.run_cfg.pred_algo.lower() == 'sdae':
            supervised_model = self.dense_feats.model
            
            n_layers, n_hid, dropout = param_config_utils.get_sdae_params(cfg, params)
            cur_sdae = SDAE(n_layers=n_layers, n_hid=n_hid, dropout=dropout)
            
            fit_model, (dense_train_ft, dense_val_ft, dense_test_ft), _ = cur_sdae.supervised_classification(model=supervised_model, x_train=self.feats.feats_train, x_val=self.feats.feats_val, y_train=self.y_train, y_val=self.y_val, x_test = self.feats.feats_test, y_test = self.y_test, n_classes=self.n_classes)
            self._write_finetuned_dense(cfg, dense_train_ft, dense_val_ft, dense_test_ft)
        
        elif cfg.run_cfg.pred_algo.lower() == 'ffnn':
            (n_hid_layers, n_hid_units, act_fn) = param_config_utils.get_ffnn_params(cfg, params)
            
            model = ffnn.get_ffnn_model(n_in=self.feats.feats_train.shape[1],
                                            n_out=self.n_classes,
                                            n_hid_layers=n_hid_layers,
                                            n_hid=n_hid_units,
                                            act_fn = act_fn
                                            )
                
            fit_model = nn_utils.train(model,
                                       x_train=self.feats.feats_train, y_train=self.y_train,
                                       x_val=self.feats.feats_val, y_val=self.y_val,
                                       n_classes=self.n_classes,
                                       nb_epoch=cfg.run_cfg.n_epochs_ffnn
                                       )
            
        else:
            model = Model(cfg.run_cfg.pred_algo, cfg.run_cfg.seed)
            fit_model = model.fit(self.feats.feats_train, self.y_train)        
        return fit_model
    
    
    def predict(self, fit_model, cfg, test_type='val'):
        """
        Returns predictions for the test dataset using trained model
        @param fit_model: trained model
        @param cfg: Config object
        @param test_type (val/test): whether to test on validation data or test data
        @return prediction probability scores
        """
        
        if cfg.run_cfg.pred_algo == 'sdae':
            cur_sdae = SDAE(n_layers=cfg.run_cfg.n_hid_layers, n_hid=cfg.run_cfg.n_hid, dropout=cfg.run_cfg.dropout, nb_epoch=cfg.run_cfg.n_epochs)
            if test_type == 'val':
                return cur_sdae.predict(fit_model, self.feats.feats_val)
            else:
                return cur_sdae.predict(fit_model, self.feats.feats_test)
        
        elif cfg.run_cfg.pred_algo.lower() == 'ffnn':
            if test_type == 'val':
                return nn_utils.predict(fit_model, self.feats.feats_val, cfg)
            else:
                return nn_utils.predict(fit_model, self.feats.feats_test, cfg)
        
        else:
            model = Model(cfg.run_cfg.pred_algo, cfg.run_cfg.seed)
            model.classifier = fit_model
            if test_type == 'val':
                return model.predict_prob(self.feats.feats_val)
            else:
                return model.predict_prob(self.feats.feats_test)
    
    def evaluate(self, y_pred, y_pred_score, test_type='val'):
        """
        Evaluates the generated predictions and returns the calculated scores and confusion matrix.
        @param y_pred: predicted labels
        @param y_pred_score: prediction probability scores
        @param test_type (val/test): whether to test on validation data or test data
        @return scores, confusion matrix
        """
        
        if test_type == 'val':
            y_true = self.y_val
        elif test_type == 'test':
            y_true = self.y_test
           
        score, conf_matrix, error_idx = model_utils.score(y_true, y_pred, y_pred_score, self.n_classes)
        
        return score, conf_matrix, error_idx
    
    def get_significant_feats(self, cfg, fit_model, pred, instance_type =  'test', output_pred = 'correct'):
        
        label_set = {'train':self.y_train, 'val': self.y_val, 'test':self.y_test}
            
        if cfg.run_cfg.pred_type in ['in_hosp','30_days','1_year','mortality']:
            indices = [i for i, x in enumerate(label_set[instance_type]) if x == 1] #get the indices for instance of a given class
        else:
            indices = [i for i, _ in enumerate(label_set[instance_type])]
        
        
        idx = self._get_instance_idx(output_pred, indices, label_set[instance_type], pred)
        print("Index for patient:", idx)
      
        print("Patient ID for which features have been extracted: ", self.patients[self.test_idx[indices[idx]]].pid)
        print("True label for patient: ", label_set[instance_type][indices[idx]])   
        print("Predicted label for patient: ", pred[indices[idx]])
                      
        if cfg.run_cfg.pretrain and cfg.run_cfg.rep_type == 'sdae':
            or_feats_test = utils.read_data(cfg.path_cfg.PATH_INPUT, cfg.run_cfg.feat_type + '_' + cfg.run_cfg.content_type + '_' + cfg.run_cfg.val_type + '_' + cfg.run_cfg.feat_level + '_'+instance_type+'.pkl.gz', pickle_data=True, compress=True)
            _, or_feats_test = self._update_data_on_class(pt_idx = self.test_idx, feats = or_feats_test, f_type = cfg.run_cfg.pred_type)
            
            
            input_significance, significance_ip_op = get_sdae_input_significance(classifier_model = fit_model, sdae_model = self.dense_feats.model, feats_classifier = self.feats.feats_test[indices[idx],:], feats_sdae = or_feats_test[indices[idx],:])
#                 utils.write_data(sensitivity_output_input_inst, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'sensitivity_input_output_sdae_'+cfg.run_cfg.pred_type+'.pkl.gz', pickle_data = True, compress = True)
            utils.write_feat_importance(input_significance, 'significance_sdae', cfg.path_cfg.PATH_OUTPUT, cfg.run_cfg.pred_type, self.feats.feat_vocab_idx, vocab_link = True)
#                 utils.write_feat_importance_class(significance_ip_op, 'significance_sdae', cfg.path_cfg.PATH_OUTPUT, cfg.run_cfg.pred_type, self.feats.feat_vocab_idx, vocab_link = True)
        else:
            input_significance, significance_ip_op = get_ffnn_input_significance(fit_model, self.feats.feats_test[indices[idx],:])
#                 utils.write_data(sensitivity_output_input_inst, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'sensitivity_input_output_ffnn_'+cfg.run_cfg.pred_type+'.pkl.gz', pickle_data = True, compress = True)
            utils.write_feat_importance(input_significance, 'significance_classifier', cfg.path_cfg.PATH_OUTPUT, cfg.run_cfg.pred_type, self.feats.feat_vocab_idx, vocab_link = True)
#                 utils.write_feat_importance_class(significance_ip_op, 'significance_sdae', 'significance_classifier', cfg.path_cfg.PATH_OUTPUT, cfg.run_cfg.pred_type, self.feats.feat_vocab_idx, vocab_link = True)
    
    def _get_instance_idx(self, output_pred, indices, label_set, pred):
        return 4 
        for idx, i in enumerate(indices):
            if output_pred == 'correct' and label_set[i] == pred[i]:
                return idx
            elif output_pred == 'incorrect' and label_set[i] != pred[i]:
                return idx
        return 0
                               
    def _write_finetuned_dense(self, cfg, dense_train_ft, dense_val_ft, dense_test_ft):
        utils.write_data(dense_train_ft, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_train_ft.pkl.gz', pickle_data= True, compress = True)
        utils.write_data(dense_val_ft, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_val_ft.pkl.gz', pickle_data= True, compress = True) 
        utils.write_data(dense_test_ft, out_dir = cfg.path_cfg.PATH_OUTPUT, f_name = 'sdae_dense_test_ft.pkl.gz', pickle_data= True, compress = True) 
        
    def cluster(self, cfg):
        """
        Cluster features
        @param cfg: Config object
        """
        Cluster(data_in = self.feats.feats_train, data_test = self.feats.feats_val, labels_test = self.y_val, n_clusters= self.n_classes, algo = 'k_means')
           
    def visualize_vec(self, cfg):
        self.load_dense_feats(cfg)
        
        pids_train = [self.patients[idx].pid for idx in self.train_idx]
        visualize.plot_vectors(self.dense_feats.dense_train[:500, :], labels=pids_train[:500], label_type='pids_train_500', out_dir=cfg.path_cfg.PATH_OUTPUT, seed=cfg.run_cfg.seed)
        
        pids_val = [self.patients[idx].pid for idx in self.val_idx]
        visualize.plot_vectors(self.dense_feats.dense_val[:500, :], labels=pids_val[:500], label_type='pids_val_500', out_dir=cfg.path_cfg.PATH_OUTPUT, seed=cfg.run_cfg.seed)
        
#         self.get_labels(cfg)
#         visualize.plot_vectors(self.feats.feats_val[:200,:], labels = self.y_val[:200], label_type = 'mortality', out_dir = cfg.path_cfg.PATH_OUTPUT, seed = cfg.run_cfg.seed)
    
    def write_vis_data(self, cfg):
        """
        Write datasets in CSV format for visualization
        1. Write the dense representations in CSV
        2. Write the labels corresponding to dense representation (in corresponding row IDs except for added header in labels)
        @param cfg: Config object
        """
        
        self.load_dense_feats(cfg)
        self._write_dense_csv(cfg)
#         self._get_csv_lables(cfg)
#         self._write_pt_labels_csv(cfg)

    def _write_dense_csv(self, cfg):
        """
        Write the dense patient representation in CSV format.
        """
        self.dense_feats._write_csv(cfg)
    
    def _write_pt_labels_csv(self, cfg):
        """
        Write the labels in CSV format
        @param cfg: Config object
        """
        
        self._write_csv_labels(cfg, ds_type='train')
        self._write_csv_labels(cfg, ds_type='val')
        self._write_csv_labels(cfg, ds_type='test')
        
    def _write_csv_labels(self, cfg, ds_type):
        """
        Write the labels for a given subset of patients in csv format
        @param cfg: Config object
        @param ds_type: the type of data subset (train/val/test) to write labels for
        """
        if ds_type == 'train':
            idx = self.train_idx
        elif ds_type == 'val':
            idx = self.val_idx
        elif ds_type == 'test':
            idx = self.test_idx
            
        with gzip.open(cfg.path_cfg.PATH_INPUT + 'mimiciii_labels_' + ds_type + '.csv.gz', 'w') as fout:
                
            fout.write(b"gender,marital_status,admission_location,diagnoses,diagnostic_categories,procedure_categories,mortality_class,current_care_unit,n_med_types,n_med")
            fout.write(b'\n')
        
            for cur_idx in idx:
                
                if self.patients[cur_idx].gender is None:
                    self.patients[cur_idx].gender = ''
                    
                if self.patients[cur_idx].mar_stat is None:
                    self.patients[cur_idx].mar_stat = ''
                else:
                    self.patients[cur_idx].mar_stat = self.patients[cur_idx].mar_stat.replace(',',' ')
                    
                if self.patients[cur_idx].adm_loc is None:
                    self.patients[cur_idx].adm_loc = ''
                else:
                    self.patients[cur_idx].adm_loc = self.patients[cur_idx].adm_loc.replace(',', ' ')
                    
                if self.patients[cur_idx].diagnosis is None:
                    self.patients[cur_idx].diagnosis = ''
                else:
                    self.patients[cur_idx].diagnosis = self.patients[cur_idx].diagnosis.replace(',',';')
                    
                str_to_write = self.patients[cur_idx].gender + ',' + self.patients[cur_idx].mar_stat + ',' + self.patients[cur_idx].adm_loc + ',' + self.patients[cur_idx].diagnosis + ','
                
                for i, cur_diag_cat in enumerate(self.patients[cur_idx].diag_cats):
                    if i:
                        str_to_write += ';'
                    str_to_write = str_to_write + cur_diag_cat.replace(',', '-')
                
                str_to_write += ','
                
                for i, cur_proc_cat in enumerate(self.patients[cur_idx].proc_cats):
                    if i:
                        str_to_write += ';'
                    str_to_write = str_to_write + cur_proc_cat.replace(',', '-')       
                
                str_to_write += ','
                
                if self.patients[cur_idx].n_med_type is not None:
                    n_med_type = str(self.patients[cur_idx].n_med_type)
                else:
                    n_med_type = ''
                    
                if self.patients[cur_idx].n_med is not None:
                    n_med = str(self.patients[cur_idx].n_med)
                else:
                    n_med = ''
                              
                if self.patients[cur_idx].ccu is None:
                    self.patients[cur_idx].ccu = ''              
                              
                str_to_write += str(self.patients[cur_idx].mortality) + ',' + self.patients[cur_idx].ccu  + ',' + n_med_type + ',' + n_med
                
                fout.write(str_to_write.encode())
                fout.write(b'\n')
     
    def write_all_train_tokenized(self, cfg):
        '''
        Concatenate all the tokens in training notes (separated by space) into one file to generate word embeddings.
        @param cfg: config object
        '''
        with open(cfg.path_cfg.PATH_EMBEDDING + 'mimic_all_train_tokenized.txt', 'w') as f_out:
            for cur_idx in self.train_idx:
                for cur_note in self.patients[cur_idx].notes:
                    content = open(cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC + str(cur_note.pid) + '_' + str(cur_note.hadmid) + '_' + str(cur_note.noteid) + '.txt').read()
                    f_out.write(content + '\n')
    
    def get_pt_stats(self, cfg):
        
        print("Getting patient statistics about n_notes and n_tokens")
        with open(cfg.path_cfg.PATH_OUTPUT+'pt_note_stats.csv','w') as f:
            f.write('S.No.,Patient ID,n_notes,n_tokens\n')
            for i, cur_pt in enumerate(self.patients):
                print("Instance: ",i)
                cur_token_len = 0
                for cur_note in cur_pt.notes:
                    f_note = open(cfg.path_cfg.PATH_PROCESSED_UCTO_MIMIC+str(cur_note.pid) + '_' + str(cur_note.hadmid) + '_' + str(cur_note.noteid) + '.txt')
                    cur_token_len += len(f_note.read().split())
                f.write(str(i)+','+str(cur_pt.pid)+','+str(len(cur_pt.notes))+','+str(cur_token_len)+"\n")
        
                
    def get_text_stats(self, cfg):
        '''
        Prints the statistics about the minimum, maximum, and the average number of words across sentences, 
            sentences across notes, 
            and notes across patients in the training data
        Also prints the percentage of patients having up to 30, 50, 80, and 100 notes
        '''
        
        n_pt = len(self.train_idx)
        min_notes = sys.maxsize
        max_notes = tot_notes = pt_30_notes = pt_50_notes = pt_80_notes = pt_100_notes = 0.0
        
        min_note_len = sys.maxsize
        tot_note_len = max_note_len = n_notes = 0.0
        
        min_sent_len = sys.maxsize
        tot_sent_len = max_sent_len = tot_sent = 0.0
        
        for cur_idx in self.train_idx:
            
            n_notes = len(self.patients[cur_idx].notes)
            
            if n_notes < min_notes:
                min_notes = n_notes
            if n_notes > max_notes:
                max_notes = n_notes
                
            tot_notes += n_notes
            
            if n_notes <= 30:
                pt_30_notes += 1
            if n_notes <= 50:
                pt_50_notes += 1
            if n_notes <= 80:
                pt_80_notes += 1
            if n_notes <= 100:
                pt_100_notes += 1
            
            for cur_note in self.patients[cur_idx].notes:
                cur_note_len = 0
                with open(cfg.path_cfg.PATH_CONTENT_MIMIC + str(cur_note.pid) + '_' + str(cur_note.hadmid) + '_' + str(cur_note.noteid) + '.txt') as f_note:
                    for cur_sent in f_note:
                        cur_sent = cur_sent.strip()
                        
                        if cur_sent != '':  # count the sentence only if it is non empty
                            cur_note_len += 1
                            tot_sent += 1
                        
                            cur_sent_len = len(cur_sent.split())
                            
                            tot_sent_len += cur_sent_len
                            
                            if cur_sent_len < min_sent_len:
                                min_sent_len = cur_sent_len
                            if cur_sent_len > max_sent_len:
                                max_sent_len = cur_sent_len
                                
                    tot_note_len += cur_note_len
                    if cur_note_len < min_note_len:
                        min_note_len = cur_note_len
                    if cur_note_len > max_note_len:
                        max_note_len = cur_note_len
                
        print("Minimum no. of notes for a patient: ", min_notes)
        print("Maximum no. of notes for a patient:", max_notes)
        print("Average no. of notes per patient", tot_notes / n_pt)
                
        print("Percentage of patients having up to 30 notes", pt_30_notes / n_pt * 100) 
        print("Percentage of patients having up to 50 notes", pt_50_notes / n_pt * 100)
        print("Percentage of patients having up to 80 notes", pt_80_notes / n_pt * 100)        
        print("Percentage of patients having up to 100 notes", pt_100_notes / n_pt * 100)     
        
        print("Minimum no. of sentences in a note: ", min_note_len)
        print("Maximum no. of sentences in a note: ", max_note_len)
        print("Average no. of sentences in a note: ", tot_note_len / tot_notes)
        
        print("Minimum no. of tokens in a sentence: ", min_sent_len)
        print("Maximum no. of tokens in a sentence: ", max_sent_len)
        print("Average no. of tokens in a sentence: ", tot_sent_len / tot_sent)
    
#     def write_datasets(self, cfg):
#         '''
#         Write the datasets for four types of mortality:
#         1) If the patient died in hospital: binary
#         2) If the patient died 30 days after discharge: binary
#         3) If the patient died 1 year after discharge: binary
#         4) Multiclass with the classes being 1) 2) and 3) or 0) for none of the others
#         @param cfg: config file
#         '''
#         self._write_mort_hosp(cfg)
#         self._write_mort_30_days(cfg)
#         self._write_mort_1_year(cfg)
#         self._write_mort_all(cfg)
#         
#     
#     def _write_mort_hosp(self, cfg):
#         '''
#         Write dataset to check if the patient died within the hospital
#         '''
#         self.get_in_hosp_mortality()
#         utils.write_dataset_to_file(out_dir = cfg.path_cfg.PATH_INPUT, 
#                                     f_name = 'mimic_mortality_hosp.pkl.gz', 
#                                     x_train = self.feats.feats_train, y_train = self.y_train, 
#                                     x_val = self.feats.feats_val, y_val = self.y_val, 
#                                     x_test = self.feats.feats_test, y_test = self.y_test)
#         
#     def _write_mort_30_days(self, cfg):
#         '''
#         Write dataset to check if the patient died within 30 days after discharge
#         '''
#         self.get_30_days_mortality()
#         
#         utils.write_dataset_to_file(out_dir = cfg.path_cfg.PATH_INPUT, 
#                                     f_name = 'mimic_mortality_30_days.pkl.gz', 
#                                     x_train = self.feats.feats_train, y_train = self.y_train, 
#                                     x_val = self.feats.feats_val, y_val = self.y_val, 
#                                     x_test = self.feats.feats_test, y_test = self.y_test)
#         
#     def _write_mort_1_year(self, cfg):
#         '''
#         Write dataset to check if the patient died within 1 year after discharge
#         '''
#         self.get_1_year_mortality()
#         utils.write_dataset_to_file(out_dir = cfg.path_cfg.PATH_INPUT, 
#                                     f_name = 'mimic_mortality_1_year.pkl.gz', 
#                                     x_train = self.feats.feats_train, y_train = self.y_train, 
#                                     x_val = self.feats.feats_val, y_val = self.y_val, 
#                                     x_test = self.feats.feats_test, y_test = self.y_test)
#             
#     def _write_mort_all(self, cfg):
#         '''
#         Write dataset to check if the patient died within hospital, within 30 days after discharge, within 1 year after discharge, or none of the above.
#         '''       
#         self.get_multiclass_mortality()
#         utils.write_dataset_to_file(out_dir = cfg.path_cfg.PATH_INPUT, 
#                                     f_name = 'mimic_mortality.pkl.gz', 
#                                     x_train = self.feats.feats_train, y_train = self.y_train, 
#                                     x_val = self.feats.feats_val, y_val = self.y_val, 
#                                     x_test = self.feats.feats_test, y_test = self.y_test)
       
