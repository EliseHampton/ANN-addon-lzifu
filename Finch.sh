
## ======================== Welcome to Finch ========================= ##
# Finch is a wrapper bash script to run TheMachine
# (If you get the reference you are awesome!)
# Finch runs several programs. one after the other, to either train or to
# predict the answers using a pre-trained set of parameters. In either
# case it can be run from here.
# Finch calls both python and octave scripts (You will need both installed
# before continuing)

# Finch uses the output from fitting 1, 2, and 3 component fits with
# lzifu (Ho et al 2015)
# The idea is to make it fast and easy to work out which is the best
# fit for each spaxel.
# The initial testing data was created by checking each spaxel by eye
# Those involved in this were: Elise Hampton, Rebecca Davies, and Lisa
# Kewley. Thank you!
# This data set gave something for TheMachine to learn from.
## =================================================================== ##

## ============================= Author ============================== ##
# Elise Hampton
# PhD Candidate Australian National University
# Research Schol of Astronomy and Astrophysics
# Finch written fot the S7 collaboration (PI: A/Prof Michael Dopita) and
# the SAMI Survey (PI: Prof Scott Croom).
# Supervisor: Prof. Lisa Kewley
# Last update: January 2015
## =================================================================== ##


## =============================== log =============================== ##
# 12th Jan 2015 - Finch created
# 12th Jan 2015 - user input tested - good!
# 12th Jan 2015 - running CreateInput and theMachine from inside script
# 12th Jan 2015 - need to update input and machine to take in arguments
# 13th Jan 2015 - update user interface to read out slowly (for fun)
# 13th Jan 2015 - Now reads out strings in telletyper fashion
# 19th Jan 2015 - Sends train and test as command line arguments to
# CreateInput, TheMachine, and reimage
## =================================================================== ##
clear

START=$(date +%s.%N)

foo="Hello Elise!"
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
#echo ' '
#sleep 0.5

foo="What are we doing today?"
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
#echo ' '
#sleep 0.5
foo="Training (train)? Testing (test)? Both (traintest)? or just running (run)?"
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '

while :
do
    read tmp

    if [ "$tmp" == "run" ]; then
	train=0
	testt=0
	break
    elif [ "$tmp" == "train" ]; then
	train=1
	testt=0
	break
    elif [ "$tmp" == "test" ]; then
	foo="We can't test without training at the moment! Setting to Both."
	for (( i=0; i<${#foo}; i++ )); do
	    printf "%s" "${foo:$i:1}"
	    sleep 0.2
	done
	echo ' '
	train=1
	testt=1
	break
    elif [ "$tmp" == "traintest" ]; then
	train=1
	testt=1
	break
    else
	echo "invalid input.... Try again?"
    fi
done

echo "Calling createNanmasks.py on galaxies.txt."
printf $foo
python2.7 createNanmasks.py

#sleep 1
#call create input
foo="Calling CreateInput.py with train($train) and test($testt)."
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '
#python2.7 CreateInput.py ${train} ${testt}
python2.7 CreateInput_sami.py ${train} ${testt}

#sleep 1
#call themachine
foo="Calling TheMachine.m with train($train) and test($testt)."
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '
START2=$(date +%s.%N)
/Applications/Octave.app/Contents/Resources/bin/octave --silent --eval "theMachine" ${train} ${testt}
#/Applications/Octave.app/Contents/Resources/bin/octave --silent --eval "theMachine"
#octave --silent --eval "theMachine"

#sleep 1
#call reimage
foo="Calling reimage.py train($train) and test($testt)."
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '
#python2.7 reimage.py ${train} ${testt}
python2.7 reimage_sami.py ${train} ${testt}
END2=$(date +%s.%N)
DIFF2=$END2-$START2
echo ${DIFF2}
#sleep 1
#call merge comp
foo="Calling merge_components.py"
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '
#python2.7 merge_comp.py

#sleep 1
foo="Task Complete. See you next time!"
echo ${foo}
#for (( i=0; i<${#foo}; i++ )); do
#    printf "%s" "${foo:$i:1}"
#    sleep 0.2
#done
echo ' '
#sleep 1

END=$(date +%s.%N)
DIFF=$END-$START
echo ${DIFF}
