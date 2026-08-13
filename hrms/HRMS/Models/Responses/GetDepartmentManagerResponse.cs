namespace HRMS.Models.Responses;

public class GetDepartmentManagerResponse
{
    public int Id { get; set; }
    public string Username { get; set; }
    
    public string SlackId { get; set; }
    
    public string SlackUsername { get; set; }    

    public string SlackRealName { get; set; }
}