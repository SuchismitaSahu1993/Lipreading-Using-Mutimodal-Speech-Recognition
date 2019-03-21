'''
This script downloads the TCD-TIMIT database to the current directory,
creating a proper directory structure
HOW TO USE:
1. Login to the TCD- TIMIT website
2. Download the firefox plugin 'cliget'
3. Go to some file on the TCD- TIMIT downloads page, right click
4. 'copy curl for link' in the cliget menu
5. Get the 'Cookie' header,
6. Replace the CH  in this script with it
'''
import os.path as path
import os
import sys
import pycurl

parent_dir =  path.abspath(path.join(__file__ ,"../../..")) 
CH = "SSESSa08f1a9d41786c65667603b759c65eb0=U6GxZC3pJvaxJFPGV3CO70fVSM4tLPdnawR0TjQNOs4"
COOKIE_HEADER = "has_js=1; " + CH
print("Downloading the TCD-TIMIT database to ", parent_dir)
print('#'*50)

'''
    Read the input file, formatted as:
    dirName: URL
    So for example 'Lipspkr1: https://sigmedia.tcd.ie/TCDTIMIT/filebrowser/download/1192'

    -> split each line at the colon. The part before = dir name. The part after= URL
    Create the directory if it doesn't exist yet,
    Go down into it and download the file referenced by the URL.
    loop through all the lines of the input file until everything is downloaded
'''

def progress(download_t, download_d, upload_t, upload_d):
    '''
        Callback function for pycurl to show download progress.
    '''
    if download_d and download_t:
        print('*' * int((download_d / (download_t + download_d))*100),
            round((download_d / (download_t + download_d)) * 100, 2), '% completed', end='\r')
        sys.stdout.flush()

for urls in open('downloadLocations.txt'):
    os.chdir(parent_dir)
    link = urls.split()
    for line in link:
        # for the URL, you should be cd'ed into the right place. Now download the file.
        if "https" in line:
            print("URL: ", line)
            if os.path.exists(os.getcwd()+'/straightcam.zip'): # Check if file already exists
                print("File exists. Skipping...")
            elif os.path.exists(os.getcwd()+'/straightcam'):  # Check if file already exists
                print("File exists. Skipping...")
            else:
                print("File does not exist yet. Downloading...")
                f = open('straightcam.zip', 'wb')
                C = pycurl.Curl()
                C.setopt(C.URL, line)
                C.setopt(C.COOKIE, COOKIE_HEADER)
                C.setopt(C.WRITEDATA, f)
                C.setopt(C.NOPROGRESS, False)
                C.setopt(C.XFERINFOFUNCTION, progress)
                C.perform()
                cwd = os.chdir(os.getcwd())
                print("%s downloaded" %(line))
                print("------------------------------------------")
        else:
            # For the directory, create dir structure and cd into it.
            # Next loop iteration, you'll encounter the URL
            # Download the file over there
            line = line.replace(":", "")  # Remove colon from dirname
            print("Folder name: ", line)

			# Make the directories and go in there to then be able to download the file
            if "Lipspkr" in line:
                # If lipspeaker, go top the lipspeaker top-level folder
                # Otherwise to the volunteer folder
                directory = "TCDTIMITDataset/lipspeakers/"+line+"/Clips"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cwd = os.chdir(directory)
            else:
                directory = "TCDTIMITDataset/volunteers/"+line+"/Clips"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                cwd = os.chdir(directory)
