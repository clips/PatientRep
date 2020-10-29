import config as cfg
import utils

'''
Lexicon of all MOLIS lab test description, as present in corresponding database table.
Each description is written on a new line, and spaces between different words in a description have been replaced with '_'
'''
def create_molis_lexicon(host, port, uname, pwd, db_name, tname, out_dir, lexicon_fname):
    db_conn = utils.connect_to_db(host, port, uname, pwd, db_name)
    # creating Cursor object to execute sql queries
    cur = db_conn.cursor()
        
    #forming table name for query, prefixed with database name
    table = db_name+'.'+ tname
        
    #Executing SQL queries within cursor
    query = 'select distinct descr FROM '+table
    cur.execute(query)
    
    f = open(out_dir+lexicon_fname, 'w')
    for cur_name in cur.fetchall():
        name = cur_name[0].rstrip(':')
        name = name.rstrip('.')
        f.write(name+'\n') 
        
       
if __name__ == '__main__':
    create_molis_lexicon(host = cfg.MYSQL_HOST, port = cfg.MYSQL_PORT, uname = cfg.MYSQL_UNAME, pwd = cfg.MYSQL_PASS, db_name = cfg.DB_EXT, tname = cfg.TABLE_MOLIS, out_dir = '../'+cfg.PATH_RESOURCES, lexicon_fname = 'molis_labs.txt')