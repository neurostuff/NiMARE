#!/bin/bash
#
# Collects the pull-requests since the latest release and
# aranges them in the CHANGES.rst.txt file.
#
# This is a script to be run before releasing a new version.
# Authored by Oscar Esteban
#
# Usage /bin/bash update_changes.sh 1.0.1
#
# Copyright (c) 2015-2018, the CRN developers team.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of fmriprep nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Setting      # $ help set
# set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

# Check whether the Upcoming release header is present
head -1 CHANGES.rst | grep -q Upcoming
UPCOMING=$?
if [[ "$UPCOMING" == "0" ]]; then
    head -n3  CHANGES.rst >> newchanges
fi

# Elaborate today's release header
HEADER="$1 ($(date '+%B %d, %Y'))"
echo $HEADER >> newchanges
echo $( printf "%${#HEADER}s" | tr " " "=" ) >> newchanges
echo "" >> newchanges

# Search for PRs since previous release
git log --grep="Merge pull request" `git describe --tags --abbrev=0`..HEAD --pretty='format:  * %b %s' | sed  's/Merge pull request \#\([^\d]*\)\ from\ .*/(\#\1)/' >> newchanges
echo "" >> newchanges
echo "" >> newchanges

# Add back the Upcoming header if it was present
if [[ "$UPCOMING" == "0" ]]; then
    tail -n+4 CHANGES.rst >> newchanges
else
    cat CHANGES.rst >> newchanges
fi

# Replace old CHANGES.rst with new file
mv newchanges CHANGES.rst
