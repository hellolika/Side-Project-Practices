using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class GetAttendanceByIdRequest : ByMemberIdRequest
    {
        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }
        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }

        override
            public void CheckModel()
        {
            base.CheckModel();

            if(StartDate > EndDate)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "StartDate cannot be greater than EndDate");
            }
            
        }
    }
}
