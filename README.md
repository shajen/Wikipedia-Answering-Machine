# Wikipedia Answering Machine

This project contains my Master Thesis work about answering wikipedia questions automatically. It uses Polish Wikipedia, but switching to another language should take a few minutes. You just need to change the Wikipedia dump and few polish keywords.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need some tools to start the work.

```
python3 python3-pip
```

Install it before continue. For example on Debian based distribution run follow commands:
```
sudo apt-get install python3 python3-pip
```

### Repository

Clone the repository into your local machine. After that, install needed libraries.
```
pip3 install --user -r requirements.txt
```

### Run

Run below commands in terminal:
```
./01-prepare_database.sh
./02-download_files.sh
./03-xml_to_json.sh
./04-parse_articles.sh
./05-parse_questions.sh
./06-resolve.sh
./07-report.sh
```
or
```
./99-run_all.sh
```
Be patient, it may takes even a few hours or days, especially `06-resolve.sh` script.

## Contributing

In general don't be afraid to send pull request. Use the "fork-and-pull" Git workflow.

1. **Fork** the repo
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the **latest** from **upstream** before making a pull request!

## Donations

If you enjoy this project and want to thanks, please use follow link:

[![Support via PayPal](https://www.paypalobjects.com/webstatic/en_US/i/buttons/pp-acceptance-medium.png)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=shajen@shajen.pl&lc=US&item_name=Donate+Micropython+esp8266&no_note=0&cn=&curency_code=USD)

## License

[![License](https://img.shields.io/:license-GPLv3-blue.svg?style=flat-square)](https://www.gnu.org/licenses/gpl.html)

- *[GPLv3 license](https://www.gnu.org/licenses/gpl.html)*

## Acknowledgments

- *[Wiki](https://www.wikipedia.org/)*
