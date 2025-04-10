#!/bin/bash
###
# grader.sh
# Andrew M. Becker
# Georgia Institute of Technology
# (C) 2020
###

#Read the master config
CLASS_NAME=$(cat autograder/master.config | grep CLASS_NAME | cut -d ":" -f2)
LAB_NAME=$(cat autograder/master.config | grep LAB_NAME | cut -d ":" -f2)
COMPILE_POINTS=$(cat autograder/master.config | grep COMPILE_POINTS | cut -d ":" -f2)
COMPILE_TARGET=$(cat autograder/master.config | grep COMPILE_TARGET | cut -d ":" -f2)
#End read master config

#Defaults
TOTAL_POINTS=0
FILES_MODIFIED=0
TESTFILE_PATH="autograder"
#End Defaults

REDCOLOR="\033[31m\033[1m"
GREENCOLOR="\033[32m\033[1m"
YELLOWCOLOR="\033[33m\033[1m"
STDCOLOR="\033[0m"

#Check we at least have the master config
if [ "$LAB_NAME" == "" ] || [ "$COMPILE_POINTS" == "" ]
then
    echo -e "${REDCOLOR}Grader not configured correctly"
    exit 0
fi
#End check

echo -e "${GREENCOLOR}--------------------------------${STDCOLOR}"
echo -e "${GREENCOLOR}* $CLASS_NAME Autograder - $LAB_NAME${STDCOLOR}"
echo -e "${GREENCOLOR}--------------------------------${STDCOLOR}"
if [ "$1" != "" ] && [ -d "$1" ]
then
    echo -e "${YELLOWCOLOR}* Overriding test file directory with: [$1]${STDCOLOR}"
    TESTFILE_PATH="$1"
elif [ "$1" != "" ]
then
    echo -e "${REDCOLOR}* Tried to override test directory but [$1] does not exist${STDCOLOR}"
    echo -e "${REDCOLOR}* Reverting to default [$TESTFILE_PATH]${STDCOLOR}"
fi
echo -e "${GREENCOLOR}* Checking provided file consistency${STDCOLOR}"
while read line
do
    filename=$(echo $line | cut -d " " -f2)
    csum=$(echo $line | cut -d " " -f1)

    echo -n "Checking [$filename]..."
    cval=$(md5sum $filename | cut -d " " -f1)
    if [ "$csum" == "$cval" ]
    then
        echo -e "${GREENCOLOR}PASSES${STDCOLOR}"
    else
        echo -e "${REDCOLOR}FAIL${STDCOLOR}"
        FILES_MODIFIED=1
    fi
done < autograder/CHECKSUMS
if [ $COMPILE_POINTS -ne 0 ]
then
    echo -e "${GREENCOLOR}* Building solution${STDCOLOR}"
    make clean
    make $COMPILE_TARGET
    buildres=$?
    echo -e "${GREENCOLOR}* Checking status of compilation${STDCOLOR}"
    if [ $buildres -eq 0 ]
    then
        echo -e "${GREENCOLOR}=> Build successful - [+${COMPILE_POINTS} points]${STDCOLOR}"
        TOTAL_POINTS=$(($TOTAL_POINTS + $COMPILE_POINTS))
    else
        echo -e "${REDCOLOR}=> Build failed [+0 points]${STDCOLOR}"
    fi
fi
echo "* Running solution"
for test_routine in $TESTFILE_PATH/*.test
do
    test_name=$(cat $test_routine | grep NAME | cut -d ":" -f2)
    test_type=$(cat $test_routine | grep TYPE | cut -d ":" -f2)
    points=$(cat $test_routine | grep POINTVAL | cut -d ":" -f2)
    input_param=$(cat $test_routine | grep INPUT | cut -d":" -f2)
    binary=$(cat $test_routine | grep BINARY | cut -d":" -f2)
    #Optional
    compare_exclude=$(cat $test_routine | grep COMPARE_EXCLUDE | cut -d":" -f2)

    result=0
    echo -en "${YELLOWCOLOR}=> [$test_name] Running [$binary] with input: $input_param${STDCOLOR}\n"
    if [ "$test_type" == "internal" ]
    then
        echo -en "${YELLOWCOLOR}"
        ./$binary $input_param
        result=$?
        echo -en "${STDCOLOR}"
    elif [ "$test_type" == "mpi" ]
    then
        echo -en "${YELLOWCOLOR}"
        mpirun $input_param
        result=$?
        echo -en "${STDCOLOR}"
    elif [ "$test_type" == "compile" ]
    then
        echo -en "${YELLOWCOLOR}"
        make clean
        make $input_param
        result=$?
        echo -en "${STDCOLOR}"
    elif [ "$test_type" == "compare" ]
    then
        echo -en "${YELLOWCOLOR}"
        ./$binary $input_param > $test_routine.student
        echo -en "${STDCOLOR}"
        if [ "$compare_exclude" != "" ]
        then
            mv $test_routine.student $test_routine.student.tmp
            cat $test_routine.student.tmp | egrep -v "$compare_exclude" > $test_routine.student
            rm $test_routine.student.tmp
        fi
        echo -en "${REDCOLOR}"
        diff $test_routine.student $test_routine.ref
        result=$?
        echo -en "${STDCOLOR}"
    else
        echo -e "${REDCOLOR}=> Unrecognized Test Type${STDCOLOR}"
        result=1
    fi
    if [ $result -eq 0 ]
    then
        echo -e "${GREENCOLOR}=> [$test_name] PASSED - Awarded [+${points} points]${STDCOLOR}"
        TOTAL_POINTS=$(($TOTAL_POINTS + $points))
    else
        echo -e "${REDCOLOR}=> [$test_name] FAILED - Awarded [+0 points]${STDCOLOR}"
    fi
done
echo -e "--------------------------------"
if [ $FILES_MODIFIED -eq 0 ]
then
    echo -e "${GREENCOLOR}Total Score: $TOTAL_POINTS${STDCOLOR}"
else
    echo -e "${REDCOLOR}Total Score: $TOTAL_POINTS =>WARNING FILE MODIFICATION DETECTED - POTENTIAL 0<=${STDCOLOR}"
    TOTAL_POINTS=0
fi
exit $TOTAL_POINTS
