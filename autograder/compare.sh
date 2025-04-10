#!/bin/bash
##$1 == student
##$2 == reference
diff $1 $2 1>/dev/null 2>&1
exit $?
