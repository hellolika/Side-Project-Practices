using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Services.Interfaces;
using Newtonsoft.Json;


namespace HRMS.Models.Requests
{
    public class TakeLeaveRequest
    {
        [JsonProperty("RequestId")]
        public decimal RequestId { get; set; } = 0;
        
        [JsonProperty("NumberOfDay")]
        public decimal NumberOfDay { get; set; }

        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }

        [JsonProperty("LeaveType")]
        public int LeaveType { get; set; }
        [JsonProperty("ImagePath")] public string ImagePath { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }

        [JsonIgnore]
        public int MemberId { get; set; }

        public void CheckNumberOfDays(decimal numberOfLeaveDays)
        {
            if(NumberOfDay != numberOfLeaveDays)
            {
                ThrowException("Number of days provided does not match the date range");
            }
        }

        public void CheckModels(bool checkReason = true)
        {
            if (string.IsNullOrWhiteSpace(Reason) && checkReason)
            {
                ThrowException("Please provide a reason for the leave request");
            }
            if (!string.IsNullOrWhiteSpace(Reason) && Reason.Length > 500)
            {
                ThrowException("Maximum characters allowed for reason is 500 only");
            }
            if (StartDate > EndDate)
            {
                ThrowException("Start date cannot be greater than end date");
            }
        }

        private static void ThrowException(string message)
        {
            throw new ApiException(ApiErrorEnum.InvalidModelState, message);
        }
    }
}