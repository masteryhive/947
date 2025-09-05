
class ErrorMessages:
    AUTHENTICATION_REQUIRED = "Failed to provide assistance, you must be authenticated to access this functionality."
    CATEGORY_REQUIRED = "Failed to provide assistance, you must provide a name of the collection this data pertains to."
    MERCHANT_REQUIRED = "Failed to provide assistance, you must provide the merchant to perform this operation in the client or supplier."
    LOCATION_REQUIRED = "Failed to provide assistance, you must provide the location to perform this operation in the client."
    QUESTION_REQUIRED = "Failed provide assistance, you must give us a question."
    TOKEN_ID_REQUIRED = "Failed to generate ticket ID"
    JSON_PARSE_ERROR = "Failed to parse JSON response from the server."
    QUERY_ERROR = "An error occurred while querying vector DB.{error}"
    REPORT_TYPE_ERROR = "Invalid report type: {report_type}"
    HANDLE_ERROR = "An error occurred. {error}"
    BAD_REQUEST = "Please provide the required information in order to proceed. {error}"