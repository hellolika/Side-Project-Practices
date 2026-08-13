using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class SubmitFormBaseResponse : RepositoryBaseResponse
    {
        [JsonProperty("RequestId")]
        public int RequestId { get; set; }
    }
}
