from pytracking.evaluation.environment import EnvSettings
from local_path import base_path
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = base_path + 'pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = '/home/daikenan/dataset/OTB'
    settings.results_path = base_path + 'pytracking/'    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''

    return settings

