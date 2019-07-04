## Lesson 5
To protect privacy, the main strategy is adding noise to database 
that means
- Add random noise to the database and quires

### Privacy

**Local differential privacy** adds noise to each individual 
data point.
That means add noise directly to the database or even add noise before
putting into database. And this is the most protected approach

**Global differential privacy** adds noise to the output of the 
query on the database.
That means database contains all the private information and noise is 
applied in the interface layer to protect the privacy.

#### Local vs global
if the database operator is trustworthy,

**global differential privacy** leads more accurate results 
with the same level of privacy protection.
Namely, that the database owner should add noise properly. 


In differential privacy literature, the database owner is called 
a trusted curator.

### Local Differential Privacy
Local differential privacy is where given a collection of individuals,
each individual adds noise to their data before 
sending it to the statistical database itself. 
So everything that gets thrown into the database is already noised and
protection is happening at a local level. 
**How much noise should we add?**

**Note:** Differential privacy always requires a form of randomness 
or noise added to the query to protect from things like a 
differencing attack.

**Randomized response** is this really amazing 
technique that is used in the social sciences when trying to 
learn about the high-level trends for some taboo behavior.

**Plausible deniability**
- Flip a coin two times
- If the first coint flip is heads answer(yes/no) honestly
- If it is tails then answer according to second flip


#### Goals Differential privacy 
- To get the most accurate query results with the greatest 
amount of privacy, 
- The second goal is derivative of this, which looks at 
who trusts or doesn't trust each other in a real world situation.

If users trust each other, noise was wasted and queries are less
accurate in other case if users don't trust them each others and you
forget to add noise you put them in risk

So minimize a noise and accuracy is a tradeoff.

### how to choose
if your data set is small then you should go through 
Global differential privacy. For larger dataset use local 
differential privacy.
