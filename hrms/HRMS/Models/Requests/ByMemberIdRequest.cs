using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class ByMemberIdRequest
    {
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        public virtual void CheckModel()
        {
            if(MemberId < 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "MemberId cannot be nagative");
            }
        }
    }
}
