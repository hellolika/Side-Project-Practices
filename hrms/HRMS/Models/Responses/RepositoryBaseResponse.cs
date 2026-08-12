using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class RepositoryBaseResponse
    {
        [JsonProperty("ErrorCode")]
        public virtual ApiErrorEnum ErrorCode { get; set; }

        public void CheckErrorCode()
        {
            if(ErrorCode != ApiErrorEnum.NoError)
            {
                throw new ApiException(ErrorCode, ErrorCode.GetDescription());
            }
        }
    }
}
