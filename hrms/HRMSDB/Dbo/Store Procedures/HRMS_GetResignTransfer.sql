CREATE PROCEDURE [dbo].[HRMS_GetResignTransfer.1.0.0]
	@memberId INT,
    @startDate Date,
    @resignDate Date
AS
BEGIN

	SET NOCOUNT ON;  	
	DECLARE @dayDiff AS INT
    SELECT @dayDiff =  DATEDIFF(day,@startDate,@resignDate) + 1
    
    SELECT
    m.[Id] as MemberId,
    m.[Username],
    m.[Email], 
    m.[BankAccount],
    m.[BankName],
    m.[salary] as Salary,
    @dayDiff AS WorkingDay,
    ROUND((m.Salary/30) * @dayDiff,2) as PaySalary,
    l.NumberOfDay AS UnpaidLeave, 
    ROUND((CASE WHEN l.Amount is Null THEN 0 ElSE l.Amount END) + (CASE WHEN @dayDiff > 30 THEN 0 ELSE (m.Salary - ((m.Salary/30) * @dayDiff)) END),2) as DeductionAmount,  
    ROUND(((m.Salary/30) * @dayDiff - (CASE WHEN l.Amount is Null THEN 0 ELSE l.Amount END)),2) as NetSalary
    FROM [dbo].[Member] m
    LEFT JOIN (SELECT m.Id, a.NumberOfDay, (m.Salary/30) * a.NumberOfDay AS Amount
    FROM (
    SELECT MemberId , LeaveType , SUM(NumberOfDay)  AS NumberOfDay
        FROM TakeLeaveRecord WITH (NOLOCK)
        where MONTH(GETDATE()) = MONTH(StartDate) AND  YEAR(@resignDate) = YEAR(StartDate) AND MONTH(GETDATE()) = MONTH(EndDate) AND YEAR(GETDATE()) = YEAR(EndDate)
        GROUP BY MemberId , LeaveType
    ) a LEFT JOIN Member AS m WITH (NOLOCK)
        ON a.MemberId = m.Id 
    WHERE a.LeaveType = 0) AS l ON l.Id = @memberId
    WHERE m.[Id] = @memberId

END