#!/bin/sh

sort -u $1|grep "[A-Za-z0-9]" > $1.uniq

# Split in small chunks: 3000 pages max
split $1.uniq -l 3000 --additional-suffix=.chunks

mkdir -p dws

alias urlencode='python3 -c "import sys, urllib.parse as ul; \
 f=open(sys.argv[1]); \
 print(\"\".join([ul.quote_plus(l) for l in f ]))"'

# Download them
for f in `ls *.chunks`; do
	cat $f| awk  '{if(page!="") {page = page"\r\n"} page = page$0}END{print(page)}' > $f.post
	pages=$(urlencode $f.post)
	curl -X POST -b "WMF-Last-Access-Global=29-Oct-2018&WMF-Last-Access=29-Oct-2018&GeoIP=FR%3AARA%3AGrenoble%3A45.17%3A5.72%3Av4&VEE=wikitext" -H "Host: en.wikipedia.org" -H "User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:62.0) Gecko/20100101 Firefox/62.0" -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" -H "Accept-Language: en-US,en;q=0.5" -H "Accept-Encoding: gzip, deflate, br" -H "Referer: https://en.wikipedia.org/wiki/Special:Export" -H "Content-Type: application/x-www-form-urlencoded"  -H "Cookie: WMF-Last-Access-Global=29-Oct-2018; WMF-Last-Access=29-Oct-2018; GeoIP=FR:ARA:Grenoble:45.17:5.72:v4; VEE=wikitext" -H "Connection: keep-alive" -H "Upgrade-Insecure-Requests: 1" -d "catname=&curonly=1&wpDownload=1&wpEditToken=%2B%5C&title=Special%3AExport&pages=$pages" https://en.wikipedia.org/wiki/Special:Export > dws/$f.gz;
	rm -f $f.post;
	gzip -d dws/$f.gz;
done;

rm -f *.chunks $1.uniq

