"""

select the shots to be fed to the random apertures


"""
import numpy as np
from hetdex_api.config import HDRconfig
from hetdex_api.survey import Survey



survey_name = "hdr3"
hetdex_api_config = HDRconfig(survey_name)
survey = Survey(survey_name)
survey_table=survey.return_astropy_table()

sel_survey = (survey_table['date'] >= 20190101) & (survey_table['date'] <= 20241231) & (survey_table['fwhm_virus'] <= 2.5) & (survey_table['response_4540'] > 0.08)

np.savetxt('shots_out.txt',survey_table[sel_survey]['shotid'],fmt='%d')
