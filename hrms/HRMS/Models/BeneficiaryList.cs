using HRMS.Models.Responses;
using Newtonsoft.Json;

namespace HRMS.Models;

public class BeneficiaryList
{
    [JsonProperty("Id")]
    public int Id { get; set; }
    
    [JsonProperty("Name")]
    public string Name { get; set; }
    
    [JsonProperty("Types")]
    public List<TransferTypeResponse> BeneficiaryTypes { get; set; }
}