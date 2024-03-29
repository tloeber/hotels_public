# hotels_public

Public repository for hotels project, which pursues several different approaches to model hotel property prices. This work is based on my work as a PhD student in political science at UNC Chapel Hill, where I conducted research on addressing several long-standing methodological challenges in the social sciences. In discussions with several domain experts, I realized that the behavior of real estate prices constituted an ideal testbed for my ideas.

The project started with an attempt to use a Bayesian hierarchical model in order to model how price determinants may vary across market. While this seemed like a very appealing model in theory to model complex causality, explicitly specifying our assumptions about distributions quickly became unwieldy.  
At the same time, I had been moving more towards machine learning, and I realized that gradient boosting might be a better alternative: Let's instead use a smart algorithm that learns the complex causality from the data. The challenge was that our data set was not huge, but it turned out that gradient boosting tends to perform at least as well as regularized linear regression even for small datasets (low thousands).
