
import os
import configparser


# Read configuration file
def getConfig(filename, section, option):
    """
    :param filename file name
    :param section: Serve
    :param option: Configuration parameters
    :return: Return configuration information
    """
     
# Get the current directory path
    proDir = os.path.split(os.path.realpath(__file__))[0]
    # print(proDir)

    
# Splice paths to get the complete path
    configPath = os.path.join(proDir, filename)
    # print(configPath)

 
#Create ConfigParser object
    conf = configparser.ConfigParser()

# Read file content
    conf.read(configPath)
    config = conf.get(section, option)
    return config