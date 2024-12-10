"""

select the shots to be fed to the random apertures


"""
import numpy as np
from hetdex_api.config import HDRconfig
from hetdex_api.survey import Survey
import os.path as op
from astropy.table import Table

version = '5.0.0'
survey_name = "hdr5"
hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
survey_table=survey.return_astropy_table()

if True: #trim away those with no detections?
    #this is a big file, so it is better to run this on TACC
    catfile = op.join(hetdex_api_config.hdr_dir[survey_name], 'catalogs', 'source_catalog_' + version + '.fits')
    source_table = Table.read(catfile)

    #shots with zero detections (bad shots or not science shots) will NOT be in the source_table,
    #so just print the list of unique shots in the source_table
    #ushots, ucounts = np.unique(source_table['shotid'], return_counts=True)
    ushots = np.unique(source_table['shotid'])
    np.savetxt('shots_out.txt',ushots, fmt='%d')

    # shots_todo = []
    # for s in survey_table['shotid']:
    #     if np.count_nonzero(source_table[source_table['shotid']==s]) > 0:
    #         shots_todo.append(s)
    #
    # np.savetxt('shots_out.txt',shots_todo,fmt='%d')
else:
    #sel_survey = (survey_table['date'] >= 20180601) & (survey_table['date'] <= 20241231) & (survey_table['fwhm_virus'] <= 3.0) & (survey_table['response_4540'] > 0.08)
    #sel_survey = (survey_table['date'] >= 20200101) & (survey_table['date'] <= 20201231) & (survey_table['fwhm_virus'] <= 3.0) & (survey_table['response_4540'] > 0.08)

    #np.savetxt('shots_out.txt',survey_table[sel_survey]['shotid'],fmt='%d')
    np.savetxt('shots_out.txt', survey_table['shotid'], fmt='%d')
