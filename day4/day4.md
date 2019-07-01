### What Is Differential Privacy

Differential privacy started from 2013

**Goal:** ensure different kinds of statistical analysis don't 
compromise privacy

**Def:**
Privacy is preserved if, after the analysis, the analyzer doesn't know 
anything about the people in the dataset. They remain "unobserved".

In 1977, a reasonable definition was proposed which was this,
"Anything that can be learned from a participant in a statistical database,
can be learned without access to the database".

This definition is basically saying, anything you actually do learn 
about a person should be only public information. 
This definition assumes that information which was been made public 
elsewhere isn't harmful to an individual.

**Cynthia Dwork**
"Differential privacy" describes a promise, made by a data holder, or 
curator, to a data subject, and the promise is like this, "You will not
 be affected, adversely or otherwise, by allowing your data to be used 
 in any study or analysis, no matter what other studies, data sets, or 
 information sources, are available." 
 
 **Reference Book:** The Algorithmic Foundations of Differential Privacy
 
### Can We Just Anonymize Data
"no matter what other studies, data sets, or 
information sources, are available" this is the key issue. 

Basically, datasets are anonymized during preparation. For instance, 
Netflix published an anonymized dataset, but the user name and movie 
name have been replaced by the unique Integer.
But two researcher de-anonymized this dataset by scraping IMDB

### Introducing The Canonical Database

    import torch
    num_entries = 5000
    db = torch.rand(num_entries) > 0.5
    db
    

Imagine we have a database with just one column, so the definition of privacy in the context of this simple database.  

We performing query against the database,
If we remove a person from the database and the query doesn't change, 
then that person's privacy would be fully protected. 

In other words, if the query doesn't change after we remove someone 
from the database, then that person wasn't leaking any statistical 
information into the output of the query. 