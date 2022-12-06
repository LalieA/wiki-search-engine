from itertools import chain
from time import time
import pickle

import json
class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

CONVERGENCE_LIMIT = 0.0000001

# One click step in the "random surfer model"
# origin = probability distribution of the presence of the surfer (list of numbers) on each of the page
def surfStep(origin, links):
	dest = [0.0] * len(origin)
	for idx, proba in enumerate(origin):
		if len(links[idx]):
			w = 1.0 / len(links[idx])
		else:
			w = 0.0
		for link in links[idx]:
			dest[link] += proba * w
	return dest # proba distribution after a click


if __name__ == '__main__':
	# Running options
	DEBUG = False # prints debug values

	# Load the link information
	with open("links.dict",'rb') as f:
		links = pickle.load(f)

	t1 = time()

	# List of page titles
	allPages = list(set().union(chain(*links.values()), links.keys()))
	# For simplicity of coding we give an index to each of the pages.
	linksIdx = [ [allPages.index(target) for target in links.get(source, list())] for source in allPages ]

	# Remove redundant links (i.e. same link in the document)
	for l in links:
		links[l] = list(set(links[l]))

	# Init of the pageRank algorithm
	pageRanks = [1.0 / len(allPages)] * len(allPages) # will contain the page ranks
	delta = 1
	sourceVector = [1.0 / len(allPages)] * len(allPages) # default source
	# Or use a personalized source vector :
	# sourceVector = [(0.85) / len(allPages)] * len(allPages)

	if DEBUG:
		print("default distribution:", 1.0 / len(allPages))

	while delta > CONVERGENCE_LIMIT:
		pageRanksNew = surfStep(pageRanks, linksIdx)
		jumpProba = abs(sum(pageRanks)) - abs(sum(pageRanksNew)) # sink effect, need to jump elsewhere to prevent artificial high rank
		if jumpProba < 0: # Technical artifact due to numerical errors
			jumpProba = 0
		pageRanksNew = [ pageRank + jump for pageRank, jump in zip(pageRanksNew, (p * jumpProba for p in sourceVector)) ] # Correct for the effect describe above (jump)
		delta = abs(delta - sum([abs(pageRanksNew[n] - pageRanks[n]) for n in range(len(pageRanks))])) # Compute the delta
		pageRanks = pageRanksNew

		if DEBUG:
			print("delta:", delta, "jump proba:", jumpProba)

	# Name the entries of the pageRank vector
	pageRankDict = dict()
	for idx, pageName in enumerate(allPages):
		pageRankDict[pageName] = pageRanks[idx]
	pageRankDict = dict(sorted(pageRankDict.items(), key=lambda item: item[1], reverse=True))

	t2 = time()
	print('Done in:', t2 - t1, "seconds")

	if DEBUG:
		print("PageRank sum:", sum(pageRanks))

		with open("_pageRank.json", "w") as f :
			f.write(json.dumps(pageRankDict, indent=4, cls=SetEncoder))

		# Rank of some pages:
		print({k: pageRankDict[k] for k in list(pageRankDict)[:10]})

	# Save the ranks as pickle object
	with open("pageRank.dict",'wb') as fileout:
		pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)

