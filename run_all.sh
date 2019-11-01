#!/bin/bash

date
time bash 01-prepare_database.sh &> log/01-prepare_database.log
date
time bash 02-download_files.sh &> log/02-download_files.log
date
time bash 03-xml_to_json.sh &> log/03-xml_to_json.log
date
time bash 04-parse_articles.sh &> log/04-parse_articles.log
date
time bash 05-upload_questions.sh &> log/05-upload_questions.log
date
time bash 06-resolve.sh &> log/06-resolve.log
date
time bash 07-report.sh &> log/07-report.log
date
