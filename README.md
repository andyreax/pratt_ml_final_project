# pratt_ml_final_project
Hello! 
I'm Drey Jonathan, an MS. Library and Information Science candidate at Pratt School of information. This is a repository for my final project in INFO 656: Machine Learning. In this project I apply [Latent Direchlet Allocation] on a collection of abstracts taken from [the arxiv metadata snapshot](https://arxiv.org/). My goal was to try and identify strands of approaches/understadnings of "fairness" by machine learning practioners. 

In this readme I describe my thinking in this project and methods used. You can find datasets used in /data and my code in /notebooks. I hope to continue developing this project and asking these questions. 


### Where this Project Came From:
It’s becoming mainstream conversation that harm can be and has been done via machine learning enabled systems. Many critical practitioners, researchers, community organizers, and artists have long been calling for an end to ideas of inherent technological objectivity and an intentional reckoning with the social impacts of technology (while recogninzing that the distance between the "social" and the "technological" is a constructed one). We are moving away from traditional fears of ai (i.e. fears of generalized artificial intelligence usurping "human" "supremacy"), and looking at the [ever growing list of harms](https://datajusticelab.org/data-harm-record/) that are impacting communities right now across multiple and [intersecting axis](https://jods.mitpress.mit.edu/pub/costanza-chock/release/4) of the [matrix of domination](http://www.oregoncampuscompact.org/uploads/1/3/0/4/13042698/patricia_hill_collins_black_feminist_thought_in_the_matrix_of_domination.pdf). 

Often, those leading the fight against a harm are members of the community(s) being harmed. 

I'm coming to these conversations from critical librarianship; which centers social justice principles within the practice of librarianship, with interests in critical science and technology studies; where I look at how the ways we think about data and data-centric technologies influence how we approach them, regulate them and what kinds of relationships we build with and through them, and my own position as Black queer librarian-in-training. 

In this course, as I was learning to build pipelines and fit models I was realizing two things. Firstly, the machine learning systems, which in my own knowledge bases, were always already [systems larger than just code](https://anatomyof.ai/), in an everyday practice were dealing in math formulas; the kind that make you break out in cold sweats when you think about your primary school math trauma (note: too many of us have primary school math trauma). Secondly, on the modeling side of the terminal we the data scientists, researchers, are making so many decisions, edits, transformation on the data we train our models with, preparing our data to be machine readable, parasable, iterable and coding our own interpretations along the way. There is a neccesary shaping of data in order to usilize machine learning tools, acknowledged but accepted as a necesary compensations in order to utilize these tools. [As  critical librarians, archivists, and digital humanists continue to discuss](https://arxiv.org/pdf/1912.10389.pdf), power is always operating in these spaces of decision making and their are limits to what we can computation can do or process about when [the violence data does and that is done with data doesn't just live in objective discrete units we can "clean" of messiness](http://writingindhf19.web.unc.edu/files/2019/10/0360057.pdf) but is intimately entangled with and inflected by systemic violence beyond the terminal. 

In these conversations around "ai ethics", "equitable ai", "fair machine learning", etc; "bias" read as a kind of call to solve for "x". "Bias" sounded like a ghost in the machine; haunting the models, the datasets they trained on, the people interpreting and operating upon that data. "Fairness" was presented as an identifiable and measurable remedy to the nuanced complexities of multiple oppresiions the term "bias" stands as proxy for. I wanted to understand what exactly we mean by "fairness" in machine learning and how is that understanding being explicitly and implicitly translated into the approaches being suggested. Is "fairness" just a matter of parity or neutrality? Is that parity a neccesary compensation? Ultimately, I want to unpack the distance between our understandings of "fairness" and the not solely technical material changes and personal interregations actual material change and [justice require](https://upfromthecracks.medium.com/on-the-moral-collapse-of-ai-ethics-791cbc7df872). 

### My question:
#### "Can I identify different strands, approaches or understandings of fairness that are at play in machine learning discourse?"

### The dataset: 
[The arXiv](https://arxiv.org/) is a free open-access repository of pre peer review scholarly articles (known as pre-prints) in several STEM and STEM adjacent fields including physics, engineering and computer science hosted by Cornell University.

I chose arxiv as my data source because of it's wide use and easily available data in the form of [a metadata snapshot](https://www.kaggle.com/Cornell-University/arxiv?select=arxiv-metadata-oai-snapshot.json). 

To create my initial dataset I used the following rules: 
- A paper needed to have "fair" in the title
- A paper needed to have category tags for Machine Learning (cs.lg) and/or Artificial Intelligence (cs.ai)


### Methods:

My main dataset, made from the metadata snapshot consisted of 218 papers. I primarily worked with the full text abstracts of each paper though I used titles as tests for each step in the process. 

**Topic Modeling** is the proces a statistical method for identifying topics within a set of documents. In topic modeling   

One method of topic modelig is **Latent Dirlecht Allocation (LDA)**. LDA is an unsupervised machine learning technique which takes some number of possible topics(the "k" value), reads a set of documents, attempts to see the distribution of topics across documents, and the probability of words occuring in each topic.  

We can understand a "document" as made up of a distribution of topics, and a "topic" as made up of a collection of words. For example: In a hypothetical article about biology we might understand the words "flower", "roots" , "trees" to be major terms in the "Plants" topic. While "bees", "insects", and "hummingbird may refer to the "Animals" topic.

Data Cleaning and Normalization steps I took included:
- making all text lower case
- dropping numbers, punctuations and stop words (words with little to no meaning on their own)
- removing whitespace
- Tokenization: separating strings of text into singular terms called tokens 
- Setting N-grams: An N-gram is any sequence of n words. One words would be a unigram, two words a bi-gram, etc. I used unigrams and bi-grams that occured more than once in my text data. 
- Lemmatization: is the process of reducing a word to it's meaningfull root, called a "lemma". I.e. "learning" becomes "learn"

Data Preprocessing steps I took inclued:
- I used the [gensim library](https://radimrehurek.com/gensim/) to creating a dictionary where every token had an id number

-I used CountVectorizer() from the [sci-kit learn library](https://scikit-learn.org/stable/index.html) to create a Document-Term Matrix where the index (rows) were all documents in my dataset, the columns were all tokens from the abstracts in my dataset. 

I used the [gensim library for LDA](https://radimrehurek.com/gensim/models/ldamodel.html?highlight=ldamodel#gensim.models.ldamodel.LdaModel). 

The jupyter notebook with all of these steps is in /notebook

Important caveats are: 
- Documents have to be input to the LDA model in a particular form; as a **bag of words** where order of the words no longer matters and "non-meaningful" data like stop-words and punctuations is removed. LDA also needs the count of occurences across documents for every word in the set of documents. 

- LDA outputs whatever number of topics you asked for (k value) as collection of terms and the likelihood of a term being used in that topic. Interpretation of these word clusters and the topics they may represent is an iterative process that falls on the modeler. 


### Where I Am Now:
I tried several K values including but in k=3 I felt I began seeing possible relationships. I hope to refine k in future iterations of this project but for now my (possible) topics are:

Topic 1: 
0.031*"deep" 
0.031*"commerce" 
0.024*"attitudes" 
0.021*"cut" 
0.021*"non"
0.021*"welfare" 
0.020*"public" 
0.020*"nash" 
0.018*"claudette" 
0.017*"exchange" 

Topic 2: 
0.028*"embrace" 
0.027*"intersectional" 
0.024*"critically" 
0.024*"multiple" 
0.023*"model" 
0.022*"application" 
0.022*"allocate" 
0.019*"causal_model"
0.019*"toolkit" 
0.018*"analysis" 

Topic 3: 
0.030*"judgement" 
0.026*"responsible"
0.025*"service"
0.024*"un" 
0.021*"predictive" 
0.020*"goals" 
0.019*"group"
0.019*"counterfactual" 
0.018*"query" 
0.018*"fair_division" 

**While further analysis is neccesary before making any claim of particular findings** I can begin to infer some patterns worth going back over. I see what we might call technical termiology "model", "machine_learning", "regression" across topics (this was tru at higher k values as well). I wonder what sorts of relationships are being made between these technical terms and the terms that really stand out to me like "intersectional". Currently Topic 1 seems to have some connection to legal, policy, economic perspective/public institutions evidenced by terms like "commerce", "welfare", and ["claudette"](http://claudette.eui.eu/). Topic 2 seems to involve qualitative analysis and theories from outside of machine learning as evidenced by terms like "intersectional", "application", "model" and "embrace". Topic 3 seems to be looking at fairness at the model or data level as evidenced by "judgement", "predeictive", "group", "goals" and "fair_division"

### Further Questions /Next Steps: 

I'd like to try visualizing these topics along with further analysis. It could be interesting to see if I could classify previously unseen abstract data. I'm curious about expanding my initial dataset as well as comparing the topics around fairness I identify in machine learning to how "fairness" is discussed in other diciplines (i.e. how do sts scholars, sociologists, organizers discuss fairness?)
