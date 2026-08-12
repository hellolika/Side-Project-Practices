using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class RequestFormsResponse
    {
        [JsonProperty("Leaves")]
        public List<LeavesResponse> Leaves { get; set; }

        [JsonProperty("ReClocks")]
        public List<ReClocksResponse> ReClocks { get; set; }

        public RequestFormsResponse(List<LeavesResponse> leavesResponse, List<ReClocksResponse> reClocksResponse)
        {
            Leaves = leavesResponse;
            ReClocks = reClocksResponse;
        }
    }

}