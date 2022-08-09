# DX Lab Anomaly Detection Demo

### What is anomaly detection?
Anomaly detection (aka outlier analysis) is a step in data mining that identifies data points, events, and/or observations that deviate from a dataset’s normal behavior. Anomalous data can indicate critical incidents, such as a technical glitch, or potential opportunities, for instance, a change in consumer behavior. Machine learning is progressively being used to automate anomaly detection. 

### What is an anomaly?
With all the analytics programs and various management software available, it’s now easier than ever for companies to effectively measure every single aspect of business activity. This includes the operational performance of applications and infrastructure components as well as key performance indicators (KPIs) that evaluate the success of the organization. With millions of metrics that can be measured, companies tend to end up with quite an impressive dataset to explore the performance of their business.

Within this dataset are data patterns that represent business as usual. An unexpected change within these data patterns, or an event that does not conform to the expected data pattern, is considered an anomaly. In other words, an anomaly is a deviation from business as usual.

But then what do we mean by “business as usual” when it comes to business metrics?  Surely we don’t mean “unchanging” or “constant;” there’s nothing unusual about an eCommerce website collecting a large amount of revenue in a single day – certainly if that day is Cyber Monday. That’s not unusual because a high volume of sales on Cyber Monday is a well-established peak in the natural business cycle of any business with a web storefront.

Indeed, it would be an anomaly if such a company didn’t have high sales volume on Cyber Monday, especially if Cyber Monday sales volumes for previous years were very high. The absence of change can be an anomaly if it breaks a pattern that is normal for the data from that particular metric. Anomalies aren’t categorically good or bad, they’re just deviations from the expected value for a metric at a given point in time.

### What is time series data anomaly detection?
Successful anomaly detection hinges on an ability to accurately analyze time series data in real-time. Time series data is composed of a sequence of values over time. That means each point is typically a pair of two items — a timestamp for when the metric was measured, and the value associated with that metric at that time.

Time series data isn’t a projection in and of itself. Rather, it’s a record that contains the information necessary for making educated guesses about what can be reasonably expected in the future. Anomaly detection systems use those expectations to identify actionable signals within your data, uncovering outliers in key KPIs to alert you to key events in your organization.


# The three different types of time series anomalies
Understanding the types of outliers that an anomaly detection system can identify is essential to getting the most value from generated insights. Without knowing what you’re up against, you risk making the wrong decisions once your anomaly detection system alerts you to an issue or opportunity.

Generally speaking, anomalies in your business data fall into three main categories — global outliers, contextual outliers, and collective outliers.

### 1. Global outliers
Also known as point anomalies, these outliers exist far outside the entirety of a data set.

![img_1.png](https://2b309c36t6402pmjhp3206t0-wpengine.netdna-ssl.com/wp-content/uploads/What-is-anomaly-detection-examining-the-essentials-graph-1-1.jpg)

### 2. Contextual outliers
Also called conditional outliers, these anomalies have values that significantly deviate from the other data points that exist in the same context. An anomaly in the context of one dataset may not be an anomaly in another. These outliers are common in time series data because those datasets are records of specific quantities in a given period. The value exists within global expectations but may appear anomalous within certain seasonal data patterns.

![img_1.png](https://2b309c36t6402pmjhp3206t0-wpengine.netdna-ssl.com/wp-content/uploads/What-is-anomaly-detection-examining-the-essentials-graph-2-1.jpg)

### 3. Collective outliers
When a subset of data points within a set is anomalous to the entire dataset, those values are called collective outliers. In this category, individual values aren’t anomalous globally or contextually. You start to see these types of outliers when examining distinct time series together. Individual behavior may not deviate from the normal range in a specific time series dataset. But when combined with another time series dataset, more significant anomalies become clear.

![img_1.png](https://2b309c36t6402pmjhp3206t0-wpengine.netdna-ssl.com/wp-content/uploads/What-is-anomaly-detection-examining-the-essentials-graph-3-1.jpg)