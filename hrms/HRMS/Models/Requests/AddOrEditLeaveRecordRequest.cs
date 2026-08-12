using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{

    public class AddOrEditLeaveRecordRequest
    {

        [JsonProperty("LeaveRecordId")] public int LeaveRecordId { get; set; }

        [JsonProperty("MemberId")] public int MemberId { get; set; }

        [JsonProperty("NumberOfDay")] public decimal NumberOfDay { get; set; }

        [JsonProperty("StartDate")] public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")] public DateTimeOffset EndDate { get; set; }

        [JsonProperty("LeaveType")] public int LeaveType { get; set; }

        [JsonProperty("Reason")] public string Reason { get; set; }
        
        [JsonProperty("ImagePath")] public string ImagePath { get; set; }
        
        [JsonProperty("Status")] public int Status { get; set; }

        public void CheckModels()
        {
            
            if (LeaveRecordId < 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Leave Record Id must be greater than or equal zero.");
            }
            
            if (StartDate > EndDate)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Start date cannot be greater than end date.");
            }

            if (LeaveType < 0)
            { 
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Leave Type Id is required.");
            }

            if (Status < 0 || Status > 3 )
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Status Id is invalid.");
            }
            
        }
        
    }
}