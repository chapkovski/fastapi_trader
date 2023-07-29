_FIRST DRAFT. TO BE UPDATED AND EXPANDED_

This application is built on the [FastAPI](https://fastapi.tiangolo.com/) framework.

It uses [MongoDB](https://www.mongodb.com/) as its primary database, and [Motor](https://www.mongodb.com/docs/drivers/motor/) as a main ORM for dealing with asynchronous data access. 

However, we should consider a potential transition to a relational database in the future. One of the promising candidates for this switch is [Gino](https://python-gino.org/). For an end user (whether it is an algotrader or a real frontend user), the database we use is irrelevant because endpoints of this API remain the same. 

This API will be used by our Vue.js based frontend, but are also accessible to any client capable of making HTTP-based requests. For instance, algo-traders can interact with them,  submitting and retrievinvg data using Python `requests` (or any other similar package).


## API Endpoints 

Our API offers several endpoints, each providing unique functionality:

- `POST /sessions` - Allows the creation of new trading sessions.
- `GET /sessions/{session_id}` - Retrieves data for a specific trading session.
- `GET /traders/{trader_id}` - Retrieves data related to a specific trader.
- `GET /transactions/history` - Fetches the complete history of all transactions.

