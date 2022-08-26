#!/bin/bash/
inotifywait --monitor --timeout 7200 --event modify --include '\/[hlm].*ipynb'  ./Herron ./Lewinson ./McKinney --format "git commit -m 'Auto commit during class' %w && git push" | bash
