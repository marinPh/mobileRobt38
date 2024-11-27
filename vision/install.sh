#!/bin/bash
# ================================ HEADER =====================================
# Author:       Loic Delineau
# Date:         21/11/2024
# Licence:     	GNU-GPLv3 
# File:        	install.sh 
# Platform :    Any Ubuntu machine using apt as a package manager
# Description:	Installs all dependencies for vision of thymio

# ======================= GLOBAL VARIABLES ====================================
black=$(tput setaf 0)
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
blue=$(tput setaf 4)
magenta=$(tput setaf 5)
cyan=$(tput setaf 6)
white=$(tput setaf 7)

# ========================== FUNCTIONS ========================================
# Wait for user confirmation (y) to continue or (n) to quit                     
prompt() {                                                                      
                                                                                
        echo "Continue by entering 'y', stop the script by pressing 'n'"        
                                                                                
        # Failsafe for wrong value inputted                                     
        fail="1"                                                                
                                                                                
        while [[ "$fail" == "1" ]]; do                                          
                                                                                
                # Read 1 input char                                             
                echo -n "> "                                                    
                read -n 1 -r cmd < /dev/tty                                     
                                                                                
                # Keyboard input checking                                       
                if [[ "$cmd" == "y" ]]; then                                    
                        fail="0"                                                
                        echo ""                                                 
                elif [[ "$cmd" == "n" ]]; then                                  
                        echo ""                                                 
                        exit                                                    
                else                                                            
                        echo -e "\nUnknown key..."                              
                fi                                                              
done                                                                            
}


# ======================= USER SYSTEM IDENTIFICATION ==========================
# Checking if you are running Ubuntu
if `uname -a | grep -q ubuntu`; then
	echo "You are running Ubuntu"
	DISTRO=ubuntu
else 
	echo ""
	echo "You didn't install Ubuntu, this script won't work"
	echo "Killing script"
	echo ""
	return
	return
fi

# Checking if you have an internet connexion
if ping -c 1 -W 1 8.8.8.8 >/dev/null; then 	# this is google's DNS server
	echo "You are connected to the internet"
	WIFI=yes
else
	echo ""
	echo "You are not connected to the internet, this script won't work"
	echo "Killing script"
	echo ""
	return		# return if script is sourced or it kills shell
fi

echo ""

# =========================== SCRIPT EXECUTION ================================
# Launch prompt
echo "Starting script, it should run for about 5 minutes" # time it!!
prompt
echo ""

# Installing Packages
sudo apt install python3 -y
sudo apt install pipx -y
pipx install opencv-contrib-python --include-deps

# Not sure I should be installing this one according to tutorial on aruco?
#pipx install opencv-python --include-deps



# Installing all packages
sudo apt update && sudo apt upgrade -y

# ========================== Closing Messages ================================= 
echo "$green"
echo "Script was executed successfully! $white"
echo ""
echo "Details of this script are documented here:"
echo "https://github.com/loic-delineau/thymio"
echo ""
echo "Made,"
echo "with Love,"
echo "by Loïc"
echo ""
